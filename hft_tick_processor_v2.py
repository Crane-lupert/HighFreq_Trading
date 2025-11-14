"""
HFT 전략을 위한 고성능 코스콤 틱데이터 처리 시스템 v2
- 특정 기간 선택 전처리 추가
- 멀티파일 쿼리 명확화
- 메모리 사용량 최적화
"""

import gzip
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime
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
    def parse_filename_period(filename: str) -> Optional[Tuple[int, int]]:
        """
        파일명에서 연도와 분기/월 추출
        
        Returns:
            (year, quarter_or_month) 또는 None
        
        Examples:
            'DFKNXTRDSHRTH_2017_Q1.dat.gz' -> (2017, 1)
            'SKSNXTRDIJH_2010_02.dat.txt.gz' -> (2010, 2)
        """
        # 패턴 1: YYYY_QN (분기)
        match = re.search(r'(\d{4})_Q(\d)', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        
        # 패턴 2: YYYY_MM (월)
        match = re.search(r'(\d{4})_(\d{2})', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        
        return None
    
    @staticmethod
    def filter_files_by_period(
        file_paths: List[Path],
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        quarters: Optional[List[int]] = None,
        months: Optional[List[int]] = None
    ) -> List[Path]:
        """
        파일명 기반으로 기간 필터링
        
        Args:
            file_paths: 파일 경로 리스트
            start_year: 시작 연도 (포함)
            end_year: 종료 연도 (포함)
            quarters: 분기 리스트 [1,2,3,4]
            months: 월 리스트 [1,2,...,12]
        
        Returns:
            필터링된 파일 리스트
        
        Examples:
            # 2017~2019년 1분기만
            filter_files_by_period(files, start_year=2017, end_year=2019, quarters=[1])
            
            # 2020년 1~3월만
            filter_files_by_period(files, start_year=2020, end_year=2020, months=[1,2,3])
        """
        filtered = []
        
        for file_path in file_paths:
            period = TickDataConverter.parse_filename_period(file_path.name)
            if period is None:
                continue
            
            year, period_num = period
            
            # 연도 필터
            if start_year and year < start_year:
                continue
            if end_year and year > end_year:
                continue
            
            # 분기/월 필터
            if quarters and period_num in quarters:
                filtered.append(file_path)
            elif months and period_num in months:
                filtered.append(file_path)
            elif quarters is None and months is None:
                # 필터 없으면 연도만 체크
                filtered.append(file_path)
        
        return filtered
    
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
        filename = Path(gz_path).stem.replace('.dat', '').replace('.txt', '')
        output_path = Path(output_dir) / f"{filename}.parquet"
        
        # 이미 변환된 파일이 있으면 스킵
        if output_path.exists():
            print(f"⏭️  스킵 (이미 존재): {filename}")
            return
        
        # 필수 컬럼 선택
        cols = (TickDataConverter.DERIVATIVE_ESSENTIAL_COLS 
                if is_derivative 
                else TickDataConverter.STOCK_ESSENTIAL_COLS)
        
        print(f"변환 중: {Path(gz_path).name}")
        
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
                    df_chunk = TickDataConverter._convert_dtypes(df_chunk, is_derivative)
                    table = pa.Table.from_pandas(df_chunk)
                    
                    if writer is None:
                        schema = table.schema
                        writer = pq.ParquetWriter(output_path, schema, compression='snappy')
                    
                    writer.write_table(table)
                    chunk_data = []
                    chunk_count += 1
        
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
        
        print(f"✓ 완료: {filename}")
        print(f"  원본: {orig_size:.2f} GB → 변환: {new_size:.2f} GB (압축률: {compression_ratio:.1f}%)\n")
    
    @staticmethod
    def _convert_dtypes(df: pd.DataFrame, is_derivative: bool) -> pd.DataFrame:
        """데이터 타입 최적화"""
        numeric_cols = ['TRD_PRC', 'TRDVOL', 'OPEN_PRICE', 'HIGH_PRICE', 
                       'LOW_PRICE', 'ACC_TRDVOL', 'ACC_AMT']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'TRDVOL' in df.columns:
            df['TRDVOL'] = df['TRDVOL'].astype('int32')
        if 'ACC_TRDVOL' in df.columns:
            df['ACC_TRDVOL'] = df['ACC_TRDVOL'].astype('int64')
        if 'ACC_AMT' in df.columns:
            df['ACC_AMT'] = df['ACC_AMT'].astype('int64')
        
        if 'LST_ASKBID_TP_CD' in df.columns:
            df['LST_ASKBID_TP_CD'] = df['LST_ASKBID_TP_CD'].astype('category')
        
        if is_derivative and 'ISIN_CODE' in df.columns:
            df['ISIN_CODE'] = df['ISIN_CODE'].astype('category')
        
        return df
    
    @staticmethod
    def batch_convert(
        input_dir: str,
        output_dir: str,
        pattern: str = "*.dat.gz",
        is_derivative: bool = False,
        # 🆕 기간 필터 옵션
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        quarters: Optional[List[int]] = None,
        months: Optional[List[int]] = None
    ):
        """
        디렉토리 내 파일 일괄 변환 (기간 필터 가능)
        
        Examples:
            # 전체 변환
            batch_convert(input_dir, output_dir, is_derivative=True)
            
            # 2017~2019년만
            batch_convert(input_dir, output_dir, is_derivative=True,
                         start_year=2017, end_year=2019)
            
            # 2020년 1분기만
            batch_convert(input_dir, output_dir, is_derivative=True,
                         start_year=2020, end_year=2020, quarters=[1])
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 파일 검색
        all_files = list(Path(input_dir).glob(pattern))
        
        # 🆕 기간 필터 적용
        if start_year or end_year or quarters or months:
            filtered_files = TickDataConverter.filter_files_by_period(
                all_files, start_year, end_year, quarters, months
            )
            print(f"기간 필터 적용: {len(all_files)}개 → {len(filtered_files)}개")
        else:
            filtered_files = all_files
        
        if not filtered_files:
            print("변환할 파일이 없습니다.")
            return
        
        print(f"총 {len(filtered_files)}개 파일 변환 시작\n")
        print("="*80)
        
        for i, file_path in enumerate(filtered_files, 1):
            print(f"\n[{i}/{len(filtered_files)}] {file_path.name}")
            try:
                TickDataConverter.convert_to_parquet(
                    str(file_path),
                    output_dir,
                    is_derivative
                )
            except Exception as e:
                print(f"✗ 오류: {e}")
                continue


class MultiFileTickLoader:
    """
    🆕 여러 Parquet 파일을 동시에 쿼리
    - 파일별로 메모리에 올리지 않고 필터링
    - 필요한 데이터만 최종적으로 메모리 로드
    """
    
    @staticmethod
    def load_period_polars(
        parquet_dir: str,
        start_year: int,
        end_year: int,
        filters: Optional[Dict] = None,
        columns: Optional[List[str]] = None
    ):
        """
        특정 기간의 여러 Parquet 파일을 하나의 DataFrame으로 로드
        
        ⚠️ 주의: 메모리에 실제로 올라가는 시점은 .collect() 호출 시!
        
        Args:
            parquet_dir: Parquet 파일 디렉토리
            start_year: 시작 연도
            end_year: 종료 연도
            filters: 추가 필터 조건
            columns: 읽을 컬럼 리스트
        
        Returns:
            polars.DataFrame (메모리에 로드됨)
        
        Example:
            # 2017~2019년 KOSPI200 선물 데이터
            df = load_period_polars(
                'E:/parquet/futures',
                2017, 2019,
                filters={'ISIN_CODE': 'KR4101M30004'},
                columns=['TRD_TM', 'TRD_PRC', 'TRDVOL']
            )
        """
        import polars as pl
        
        # 파일 검색
        all_files = list(Path(parquet_dir).glob("*.parquet"))
        
        # 기간에 해당하는 파일만 필터링
        period_files = []
        for file in all_files:
            period = TickDataConverter.parse_filename_period(file.name)
            if period:
                year, _ = period
                if start_year <= year <= end_year:
                    period_files.append(file)
        
        if not period_files:
            raise ValueError(f"{start_year}~{end_year}년 데이터가 없습니다.")
        
        print(f"로딩할 파일: {len(period_files)}개")
        for f in period_files:
            print(f"  - {f.name}")
        
        # 🔑 핵심: 여러 파일을 하나의 LazyFrame으로 스캔
        # 이 시점에는 메모리 사용량 거의 0!
        df = pl.scan_parquet([str(f) for f in period_files])
        
        # 필터 적용 (아직 메모리 안 씀)
        if filters:
            for col, value in filters.items():
                if isinstance(value, list):
                    df = df.filter(pl.col(col).is_in(value))
                else:
                    df = df.filter(pl.col(col) == value)
        
        # 컬럼 선택 (아직 메모리 안 씀)
        if columns:
            df = df.select(columns)
        
        # 실행! (여기서 메모리에 로드됨)
        print(f"\n데이터 로딩 중...")
        result = df.collect()
        
        print(f"✓ 완료: {len(result):,}개 행 로드됨")
        return result
    
    @staticmethod
    def query_period_duckdb(
        parquet_dir: str,
        start_year: int,
        end_year: int,
        sql_query: str
    ):
        """
        DuckDB로 여러 파일에 SQL 쿼리 실행
        
        Args:
            parquet_dir: Parquet 파일 디렉토리
            start_year: 시작 연도
            end_year: 종료 연도
            sql_query: SQL 쿼리 (tick_data 테이블 사용)
        
        Returns:
            pandas.DataFrame
        
        Example:
            result = query_period_duckdb(
                'E:/parquet/futures',
                2017, 2019,
                '''
                SELECT ISIN_CODE, COUNT(*) as cnt, AVG(TRD_PRC) as avg_price
                FROM tick_data
                GROUP BY ISIN_CODE
                '''
            )
        """
        import duckdb
        
        # 파일 검색
        all_files = list(Path(parquet_dir).glob("*.parquet"))
        period_files = []
        
        for file in all_files:
            period = TickDataConverter.parse_filename_period(file.name)
            if period:
                year, _ = period
                if start_year <= year <= end_year:
                    period_files.append(str(file))
        
        if not period_files:
            raise ValueError(f"{start_year}~{end_year}년 데이터가 없습니다.")
        
        print(f"쿼리 대상: {len(period_files)}개 파일")
        
        con = duckdb.connect()
        
        # 여러 파일을 하나의 테이블로 읽기
        files_str = ", ".join(f"'{f}'" for f in period_files)
        query = sql_query.replace('tick_data', f'read_parquet([{files_str}])')
        
        print(f"SQL 실행 중...")
        result = con.execute(query).fetchdf()
        con.close()
        
        print(f"✓ 완료: {len(result):,}개 행 반환")
        return result


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("코스콤 틱데이터 처리 시스템 v2")
    print("="*80)
    
    # ========================================================================
    # 예시 1: 특정 기간만 전처리
    # ========================================================================
    
    print("\n\n[예시 1] 2017~2019년 1분기만 전처리")
    print("-"*80)
    
    TickDataConverter.batch_convert(
        input_dir=r"E:\선물 체결틱데이터(2010.Q1~2023.Q4)",
        output_dir=r"E:\parquet\futures",
        is_derivative=True,
        start_year=2017,
        end_year=2019,
        quarters=[1]  # 1분기만
    )
    
    # ========================================================================
    # 예시 2: 여러 파일을 한번에 쿼리 (Polars)
    # ========================================================================
    
    print("\n\n[예시 2] 2017~2019년 데이터 통합 쿼리 (Polars)")
    print("-"*80)
    
    try:
        df = MultiFileTickLoader.load_period_polars(
            parquet_dir=r"E:\parquet\futures",
            start_year=2017,
            end_year=2019,
            filters={'ISIN_CODE': 'KR4101M30004'},
            columns=['TRADE_DATE', 'TRD_TM', 'TRD_PRC', 'TRDVOL']
        )
        
        print(f"\n로드된 데이터:")
        print(df.head(10))
        print(f"\n총 {len(df):,}개 행 (메모리에 로드됨)")
        
    except Exception as e:
        print(f"Polars 예시 스킵: {e}")
    
    # ========================================================================
    # 예시 3: SQL 집계 (DuckDB)
    # ========================================================================
    
    print("\n\n[예시 3] 2017~2019년 데이터 SQL 집계 (DuckDB)")
    print("-"*80)
    
    try:
        result = MultiFileTickLoader.query_period_duckdb(
            parquet_dir=r"E:\parquet\futures",
            start_year=2017,
            end_year=2019,
            sql_query="""
                SELECT 
                    TRADE_DATE,
                    COUNT(*) as trade_count,
                    SUM(TRDVOL) as total_volume,
                    AVG(TRD_PRC) as avg_price
                FROM tick_data
                WHERE ISIN_CODE = 'KR4101M30004'
                GROUP BY TRADE_DATE
                ORDER BY TRADE_DATE
            """
        )
        
        print(f"\n집계 결과:")
        print(result.head(10))
        
    except Exception as e:
        print(f"DuckDB 예시 스킵: {e}")
    
    print("\n\n" + "="*80)
    print("완료!")
    print("="*80)
