"""
HFT 전략을 위한 고성능 코스콤 틱데이터 처리 시스템
- Parquet 변환 (80% 압축률)
- Polars/DuckDB 기반 고속 쿼리
- Numba 기반 백테스팅
"""

import gzip
import os
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


class TickDataConverter:
    """틱데이터를 Parquet로 변환 (전처리용)"""
    
    # 주식용 필수 컬럼 (HFT에 필요한 것만)
    STOCK_ESSENTIAL_COLS = {
        'TRADE_DATE': 0,
        'ISIN_CODE': 3,
        'TRD_PRC': 6,
        'TRDVOL': 7,
        'TRD_TM': 10,
        'OPEN_PRICE': 35,
        'HIGH_PRICE': 36,
        'LOW_PRICE': 37,
        'ACC_TRDVOL': 39,
        'ACC_AMT': 40,
        'LST_ASKBID_TP_CD': 41,
    }
    
    # 파생상품용 필수 컬럼
    DERIVATIVE_ESSENTIAL_COLS = {
        'TRADE_DATE': 0,
        'ISIN_CODE': 2,
        'JONG_INDEX': 3,
        'TRD_PRC': 4,
        'TRDVOL': 5,
        'TRD_TM': 8,
        'OPEN_PRICE': 11,
        'HIGH_PRICE': 12,
        'LOW_PRICE': 13,
        'ACC_TRDVOL': 15,
        'ACC_AMT': 16,
        'LST_ASKBID_TP_CD': 17,
    }
    
    @staticmethod
    def convert_to_parquet(
        gz_path: str,
        output_dir: str,
        is_derivative: bool = False,
        chunk_size: int = 1000000
    ):
        """
        GZ 파일을 Parquet로 변환 (메모리 효율적)
        
        Args:
            gz_path: 입력 .dat.gz 파일
            output_dir: 출력 디렉토리
            is_derivative: True면 선물/옵션, False면 주식
            chunk_size: 청크 크기 (기본 100만 행)
        """
        filename = Path(gz_path).stem.replace('.dat', '')
        output_path = Path(output_dir) / f"{filename}.parquet"
        
        # 필수 컬럼 선택
        cols = (TickDataConverter.DERIVATIVE_ESSENTIAL_COLS 
                if is_derivative 
                else TickDataConverter.STOCK_ESSENTIAL_COLS)
        
        print(f"변환 중: {gz_path}")
        print(f"출력: {output_path}")
        
        chunk_data = []
        chunk_count = 0
        writer = None
        schema = None
        
        with gzip.open(gz_path, 'rb') as f:
            for line_no, line in enumerate(tqdm(f, desc="읽는 중")):
                try:
                    decoded = line.decode('euc-kr').strip()
                except:
                    decoded = line.decode('cp949').strip()
                
                fields = decoded.split('|')
                
                # 필수 컬럼만 추출
                row = {name: fields[idx] if idx < len(fields) else None 
                       for name, idx in cols.items()}
                chunk_data.append(row)
                
                # 청크 단위로 저장
                if len(chunk_data) >= chunk_size:
                    df_chunk = pd.DataFrame(chunk_data)
                    
                    # 데이터 타입 변환
                    df_chunk = TickDataConverter._convert_dtypes(df_chunk, is_derivative)
                    
                    # PyArrow Table 생성
                    table = pa.Table.from_pandas(df_chunk)
                    
                    if writer is None:
                        schema = table.schema
                        writer = pq.ParquetWriter(output_path, schema, compression='snappy')
                    
                    writer.write_table(table)
                    
                    chunk_data = []
                    chunk_count += 1
                    print(f"  청크 {chunk_count} 저장 ({chunk_count * chunk_size:,} 행)")
        
        # 마지막 청크 저장
        if chunk_data:
            df_chunk = pd.DataFrame(chunk_data)
            df_chunk = TickDataConverter._convert_dtypes(df_chunk, is_derivative)
            table = pa.Table.from_pandas(df_chunk)
            
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
            
            writer.write_table(table)
        
        if writer:
            writer.close()
        
        # 파일 크기 비교
        orig_size = os.path.getsize(gz_path) / (1024**3)
        new_size = os.path.getsize(output_path) / (1024**3)
        compression_ratio = (1 - new_size/orig_size) * 100
        
        print(f"✓ 완료!")
        print(f"  원본: {orig_size:.2f} GB")
        print(f"  변환: {new_size:.2f} GB")
        print(f"  압축률: {compression_ratio:.1f}%")
    
    @staticmethod
    def _convert_dtypes(df: pd.DataFrame, is_derivative: bool) -> pd.DataFrame:
        """데이터 타입 최적화"""
        # 숫자형 변환
        numeric_cols = ['TRD_PRC', 'TRDVOL', 'OPEN_PRICE', 'HIGH_PRICE', 
                       'LOW_PRICE', 'ACC_TRDVOL', 'ACC_AMT']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 정수형 다운캐스팅
        if 'TRDVOL' in df.columns:
            df['TRDVOL'] = df['TRDVOL'].astype('int32')
        if 'ACC_TRDVOL' in df.columns:
            df['ACC_TRDVOL'] = df['ACC_TRDVOL'].astype('int64')
        if 'ACC_AMT' in df.columns:
            df['ACC_AMT'] = df['ACC_AMT'].astype('int64')
        
        # 카테고리형 변환 (메모리 절약)
        if 'LST_ASKBID_TP_CD' in df.columns:
            df['LST_ASKBID_TP_CD'] = df['LST_ASKBID_TP_CD'].astype('category')
        
        if not is_derivative:
            # 주식은 종목코드가 많아서 카테고리화하면 오히려 손해
            pass
        else:
            # 파생상품은 종목 수가 적어서 카테고리화 유리
            if 'ISIN_CODE' in df.columns:
                df['ISIN_CODE'] = df['ISIN_CODE'].astype('category')
        
        return df
    
    @staticmethod
    def batch_convert(
        input_dir: str,
        output_dir: str,
        pattern: str = "*.dat.gz",
        is_derivative: bool = False
    ):
        """디렉토리 내 모든 파일 일괄 변환"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        files = list(Path(input_dir).glob(pattern))
        print(f"총 {len(files)}개 파일 발견")
        
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {file_path.name}")
            try:
                TickDataConverter.convert_to_parquet(
                    str(file_path),
                    output_dir,
                    is_derivative
                )
            except Exception as e:
                print(f"✗ 오류: {e}")
                continue


class HighSpeedTickLoader:
    """고속 틱데이터 로더 (Polars/DuckDB 기반)"""
    
    @staticmethod
    def load_with_polars(
        parquet_path: str,
        filters: Optional[Dict] = None,
        columns: Optional[List[str]] = None
    ):
        """
        Polars로 초고속 로딩
        
        Args:
            parquet_path: Parquet 파일 경로
            filters: 필터 조건 {'ISIN_CODE': 'KR4101M30004'}
            columns: 읽을 컬럼 리스트
        
        Returns:
            polars.DataFrame
        """
        import polars as pl
        
        # Lazy evaluation
        df = pl.scan_parquet(parquet_path)
        
        # 필터 적용
        if filters:
            for col, value in filters.items():
                if isinstance(value, list):
                    df = df.filter(pl.col(col).is_in(value))
                else:
                    df = df.filter(pl.col(col) == value)
        
        # 컬럼 선택
        if columns:
            df = df.select(columns)
        
        # 실행
        return df.collect()
    
    @staticmethod
    def query_with_duckdb(
        parquet_paths: List[str],
        sql_query: str
    ):
        """
        DuckDB로 SQL 쿼리 실행 (여러 파일 동시 쿼리 가능)
        
        Args:
            parquet_paths: Parquet 파일 경로들
            sql_query: SQL 쿼리
        
        Returns:
            pandas.DataFrame
        """
        import duckdb
        
        con = duckdb.connect()
        
        # 여러 파일을 하나의 테이블처럼 쿼리
        if len(parquet_paths) == 1:
            query = sql_query.replace('tick_data', f"'{parquet_paths[0]}'")
        else:
            # 여러 파일을 UNION으로 결합
            files_str = ", ".join(f"'{p}'" for p in parquet_paths)
            query = sql_query.replace('tick_data', f"read_parquet([{files_str}])")
        
        result = con.execute(query).fetchdf()
        con.close()
        
        return result


# 사용 예시
if __name__ == "__main__":
    
    # ==================== 단계 1: GZ → Parquet 변환 (한 번만 실행) ====================
    
    print("=" * 80)
    print("단계 1: 데이터 변환 (GZ → Parquet)")
    print("=" * 80)
    
    # 선물 데이터 변환
    futures_dir = r"E:\선물 체결틱데이터(2010.Q1~2023.Q4)"
    futures_output = r"E:\parquet\futures"
    
    # 첫 파일만 테스트
    test_file = r"E:\선물 체결틱데이터(2010.Q1~2023.Q4)\DFKNXTRDSHRTH_2017_Q1.dat.gz"
    TickDataConverter.convert_to_parquet(test_file, futures_output, is_derivative=True)
    
    # 전체 변환은 주석 해제
    # TickDataConverter.batch_convert(futures_dir, futures_output, is_derivative=True)
    
    # ==================== 단계 2: 고속 쿼리 ====================
    
    print("\n" + "=" * 80)
    print("단계 2: 고속 데이터 로딩")
    print("=" * 80)
    
    # Polars 사용 예시
    try:
        import polars as pl
        
        parquet_file = Path(futures_output) / "DFKNXTRDSHRTH_2017_Q1.parquet"
        
        # 특정 종목만 필터링
        df = HighSpeedTickLoader.load_with_polars(
            str(parquet_file),
            filters={'ISIN_CODE': 'KR4101M30004'},
            columns=['TRD_TM', 'TRD_PRC', 'TRDVOL']
        )
        
        print(f"\nPolars로 로딩: {len(df):,}개 행")
        print(df.head())
        
    except ImportError:
        print("Polars 미설치: pip install polars")
    
    # DuckDB 사용 예시
    try:
        import duckdb
        
        parquet_file = Path(futures_output) / "DFKNXTRDSHRTH_2017_Q1.parquet"
        
        # SQL로 집계
        result = HighSpeedTickLoader.query_with_duckdb(
            [str(parquet_file)],
            """
            SELECT 
                ISIN_CODE,
                COUNT(*) as trade_count,
                AVG(TRD_PRC) as avg_price,
                SUM(TRDVOL) as total_volume
            FROM tick_data
            WHERE TRD_TM >= '090000000' AND TRD_TM < '100000000'
            GROUP BY ISIN_CODE
            """
        )
        
        print(f"\nDuckDB 집계 결과:")
        print(result)
        
    except ImportError:
        print("DuckDB 미설치: pip install duckdb")
    
    print("\n" + "=" * 80)
    print("완료!")
    print("=" * 80)
    print("\n다음 단계:")
    print("1. 전체 데이터를 Parquet로 변환")
    print("2. Numba로 백테스팅 전략 구현")
    print("3. Polars/DuckDB로 고속 분석")
