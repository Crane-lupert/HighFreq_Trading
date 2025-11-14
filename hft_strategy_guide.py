"""
코스콤 틱데이터 HFT 전략 구현 가이드

500GB+ 틱데이터를 효율적으로 처리하는 방법
"""

# ============================================================================
# 1단계: 환경 설정
# ============================================================================

# 필수 패키지 설치
"""
pip install pandas pyarrow polars duckdb numba tqdm
"""

# ============================================================================
# 2단계: 데이터 전처리 (GZ → Parquet)
# ============================================================================

from hft_tick_processor import TickDataConverter
from pathlib import Path

# 각 자산군별로 변환
converters = {
    '선물': {
        'input': r"E:\선물 체결틱데이터(2010.Q1~2023.Q4)",
        'output': r"E:\parquet\futures",
        'is_derivative': True
    },
    '옵션': {
        'input': r"E:\옵션 체결틱데이터(2010.Q1~2023.Q4)",
        'output': r"E:\parquet\options",
        'is_derivative': True
    },
    '유가증권': {
        'input': r"E:\유가증권 체결틱데이터(2010.1~2023.12)",
        'output': r"E:\parquet\stocks_kospi",
        'is_derivative': False
    },
    '코스닥': {
        'input': r"E:\코스닥 체결틱데이터(2010.1~2023.12)",
        'output': r"E:\parquet\stocks_kosdaq",
        'is_derivative': False
    }
}

# 일괄 변환 (시간이 오래 걸림 - 하룻밤 정도)
for name, config in converters.items():
    print(f"\n{'='*80}")
    print(f"{name} 데이터 변환 시작")
    print('='*80)
    
    TickDataConverter.batch_convert(
        config['input'],
        config['output'],
        is_derivative=config['is_derivative']
    )

# ============================================================================
# 3단계: HFT 전략 예시 - VWAP 계산
# ============================================================================

import polars as pl
from numba import jit
import numpy as np

def calculate_vwap_strategy(parquet_file: str, symbol: str):
    """
    VWAP(Volume Weighted Average Price) 전략
    
    Args:
        parquet_file: Parquet 파일 경로
        symbol: 종목코드
    """
    # Polars로 초고속 로딩
    df = pl.scan_parquet(parquet_file) \
        .filter(pl.col('ISIN_CODE') == symbol) \
        .select(['TRD_TM', 'TRD_PRC', 'TRDVOL', 'ACC_TRDVOL']) \
        .collect()
    
    # NumPy 배열로 변환 (Numba 가속용)
    prices = df['TRD_PRC'].to_numpy()
    volumes = df['TRDVOL'].to_numpy()
    times = df['TRD_TM'].to_numpy()
    
    # Numba로 VWAP 계산 (100배 빠름)
    vwap_values = calculate_vwap_numba(prices, volumes)
    
    return df.with_columns(pl.Series('VWAP', vwap_values))

@jit(nopython=True)
def calculate_vwap_numba(prices, volumes):
    """Numba 가속 VWAP 계산"""
    n = len(prices)
    vwap = np.zeros(n)
    
    cum_pv = 0.0
    cum_v = 0.0
    
    for i in range(n):
        cum_pv += prices[i] * volumes[i]
        cum_v += volumes[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0 else prices[i]
    
    return vwap


# ============================================================================
# 4단계: 백테스팅 예시 - Order Flow Imbalance
# ============================================================================

@jit(nopython=True)
def calculate_ofi(bid_volumes, ask_volumes, window=10):
    """
    Order Flow Imbalance (OFI) 계산
    
    매수/매도 압력 측정 → 단기 가격 예측
    """
    n = len(bid_volumes)
    ofi = np.zeros(n)
    
    for i in range(window, n):
        bid_sum = np.sum(bid_volumes[i-window:i])
        ask_sum = np.sum(ask_volumes[i-window:i])
        
        total = bid_sum + ask_sum
        ofi[i] = (bid_sum - ask_sum) / total if total > 0 else 0
    
    return ofi


def ofi_strategy(parquet_file: str, symbol: str, threshold=0.3):
    """
    OFI 기반 매매 신호 생성
    
    Args:
        threshold: OFI 임계값 (예: 0.3 = 매수 압력 30% 초과)
    """
    df = pl.scan_parquet(parquet_file) \
        .filter(pl.col('ISIN_CODE') == symbol) \
        .select(['TRD_TM', 'TRD_PRC', 'TRDVOL', 'LST_ASKBID_TP_CD']) \
        .collect()
    
    # 매수/매도 거래 분리
    bid_mask = (df['LST_ASKBID_TP_CD'] == '1').to_numpy()
    ask_mask = (df['LST_ASKBID_TP_CD'] == '2').to_numpy()
    
    volumes = df['TRDVOL'].to_numpy()
    bid_volumes = np.where(bid_mask, volumes, 0)
    ask_volumes = np.where(ask_mask, volumes, 0)
    
    # OFI 계산
    ofi_values = calculate_ofi(bid_volumes, ask_volumes, window=10)
    
    # 신호 생성
    signals = np.where(ofi_values > threshold, 1,  # 매수 신호
              np.where(ofi_values < -threshold, -1,  # 매도 신호
                      0))  # 중립
    
    return df.with_columns([
        pl.Series('OFI', ofi_values),
        pl.Series('SIGNAL', signals)
    ])


# ============================================================================
# 5단계: 멀티파일 분석 - DuckDB SQL
# ============================================================================

from hft_tick_processor import HighSpeedTickLoader
import duckdb

def analyze_multiple_quarters(parquet_dir: str):
    """
    여러 분기 데이터를 한 번에 분석
    
    예: 2017~2023년 전체 선물 데이터 분석
    """
    parquet_files = list(Path(parquet_dir).glob("*.parquet"))
    
    # 분기별 거래량 집계
    result = HighSpeedTickLoader.query_with_duckdb(
        [str(f) for f in parquet_files],
        """
        SELECT 
            SUBSTRING(TRADE_DATE, 1, 4) as year,
            SUBSTRING(TRADE_DATE, 5, 2) as month,
            ISIN_CODE,
            COUNT(*) as trade_count,
            SUM(TRDVOL) as total_volume,
            AVG(TRD_PRC) as avg_price,
            STDDEV(TRD_PRC) as price_volatility
        FROM tick_data
        GROUP BY year, month, ISIN_CODE
        ORDER BY year, month
        """
    )
    
    return result


# ============================================================================
# 6단계: 실전 예시 - 일중 패턴 분석
# ============================================================================

def intraday_pattern_analysis(parquet_file: str, symbol: str):
    """
    일중 시간대별 패턴 분석
    
    - 9:00-10:00: 개장 변동성
    - 11:00-12:00: 오전 마감
    - 14:00-15:00: 종가 베팅
    """
    import polars as pl
    
    df = pl.scan_parquet(parquet_file) \
        .filter(pl.col('ISIN_CODE') == symbol) \
        .with_columns([
            (pl.col('TRD_TM').cast(pl.Utf8).str.slice(0, 2).cast(pl.Int32)).alias('hour'),
            (pl.col('TRD_TM').cast(pl.Utf8).str.slice(2, 2).cast(pl.Int32)).alias('minute')
        ]) \
        .collect()
    
    # 시간대별 통계
    hourly_stats = df.group_by('hour').agg([
        pl.count().alias('trade_count'),
        pl.col('TRDVOL').sum().alias('total_volume'),
        pl.col('TRD_PRC').mean().alias('avg_price'),
        pl.col('TRD_PRC').std().alias('price_std'),
        (pl.col('TRD_PRC').max() - pl.col('TRD_PRC').min()).alias('price_range')
    ]).sort('hour')
    
    print("\n시간대별 통계:")
    print(hourly_stats)
    
    return hourly_stats


# ============================================================================
# 7단계: 성능 비교
# ============================================================================

def performance_comparison():
    """
    Pandas vs Polars vs DuckDB 성능 비교
    """
    import time
    
    # 테스트 파일
    test_file = r"E:\parquet\futures\DFKNXTRDSHRTH_2017_Q1.parquet"
    
    print("\n" + "="*80)
    print("성능 비교: 1억 행 데이터 필터링")
    print("="*80)
    
    # 1. Pandas
    start = time.time()
    df_pd = pd.read_parquet(test_file, columns=['ISIN_CODE', 'TRD_PRC'])
    df_pd = df_pd[df_pd['ISIN_CODE'] == 'KR4101M30004']
    pandas_time = time.time() - start
    print(f"Pandas: {pandas_time:.2f}초 ({len(df_pd):,}행)")
    
    # 2. Polars
    start = time.time()
    df_pl = pl.scan_parquet(test_file) \
        .filter(pl.col('ISIN_CODE') == 'KR4101M30004') \
        .select(['ISIN_CODE', 'TRD_PRC']) \
        .collect()
    polars_time = time.time() - start
    print(f"Polars: {polars_time:.2f}초 ({len(df_pl):,}행) - {pandas_time/polars_time:.1f}배 빠름")
    
    # 3. DuckDB
    start = time.time()
    con = duckdb.connect()
    df_duck = con.execute(f"""
        SELECT ISIN_CODE, TRD_PRC 
        FROM '{test_file}' 
        WHERE ISIN_CODE = 'KR4101M30004'
    """).fetchdf()
    con.close()
    duckdb_time = time.time() - start
    print(f"DuckDB: {duckdb_time:.2f}초 ({len(df_duck):,}행) - {pandas_time/duckdb_time:.1f}배 빠름")


# ============================================================================
# 실행 예시
# ============================================================================

if __name__ == "__main__":
    
    # VWAP 전략
    print("\n1. VWAP 전략 예시")
    result = calculate_vwap_strategy(
        r"E:\parquet\futures\DFKNXTRDSHRTH_2017_Q1.parquet",
        'KR4101M30004'
    )
    print(result.head(10))
    
    # OFI 전략
    print("\n2. Order Flow Imbalance 전략")
    result = ofi_strategy(
        r"E:\parquet\futures\DFKNXTRDSHRTH_2017_Q1.parquet",
        'KR4101M30004',
        threshold=0.3
    )
    print(result.filter(pl.col('SIGNAL') != 0).head(10))
    
    # 일중 패턴
    print("\n3. 일중 패턴 분석")
    hourly = intraday_pattern_analysis(
        r"E:\parquet\futures\DFKNXTRDSHRTH_2017_Q1.parquet",
        'KR4101M30004'
    )
    
    # 성능 비교
    print("\n4. 성능 비교")
    performance_comparison()
    
    print("\n" + "="*80)
    print("완료! 이제 자신만의 HFT 전략을 구현해보세요.")
    print("="*80)
