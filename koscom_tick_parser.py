"""
코스콤 체결 틱데이터 파서
공식 스펙 문서 기반
"""

import gzip
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StockTickFields:
    """유가증권/코스닥 체결 필드 정의 (45개 필드)"""
    TRADE_DATE: str = '거래일자'
    BLKTRD_TP_CD: str = '대량매매구분코드'
    REGUL_OFFHR_TP_CD: str = '정규시간외구분코드'
    ISIN_CODE: str = '종목코드'
    JONG_INDEX: str = '종목인덱스'
    TRD_NO: str = '체결번호'
    TRD_PRC: str = '체결가격'
    TRDVOL: str = '체결수량'
    TRD_TP_CD: str = '체결유형코드'
    TRD_DD: str = '체결일자'
    TRD_TM: str = '체결시각'
    NBMM_TRD_PRC: str = '근월물체결가격'
    FUTRMM_TRD_PRC: str = '원월물체결가격'
    BID_MBR_NO: str = '매수회원번호'
    BIDORD_TP_CD: str = '매수호가유형코드'
    BID_TRSTK_STAT_ID: str = '매수자사주신고서ID'
    BID_TRSTK_TRD_METHD_CD: str = '매수자사주매매방법코드'
    BID_ASK_TP_CD: str = '매수매도유형코드'
    BID_TRST_PRINC_TP_CD: str = '매수위탁자기구분코드'
    BID_TRSTCOM_NO: str = '매수위탁사번호'
    BID_PT_TP_CD: str = '매수PT구분코드'
    BID_INVST_TP_CD: str = '매수투자자구분코드'
    BID_FORNINVST_TP_CD: str = '매수외국인투자자구분코드'
    BIDORD_ACPT_NO: str = '매수호가접수번호'
    ASK_MBR_NO: str = '매도회원번호'
    ASKORD_TP_CD: str = '매도호가유형코드'
    ASK_TRSTK_STAT_ID: str = '매도자사주신고서ID'
    ASK_TRSTK_TRD_METHD_CD: str = '매도자사주매매방법코드'
    ASK_ASK_TP_CD: str = '매도매도유형코드'
    ASK_TRST_PRINC_TP_CD: str = '매도위탁자기구분코드'
    ASK_TRSTCOM_NO: str = '매도위탁사번호'
    ASK_PT_TP_CD: str = '매도PT구분코드'
    ASK_INVST_TP_CD: str = '매도투자자구분코드'
    ASK_FORNINVST_TP_CD: str = '매도외국인투자자구분코드'
    ASKORD_ACPT_NO: str = '매도호가접수번호'
    OPEN_PRICE: str = '시가'
    HIGH_PRICE: str = '고가'
    LOW_PRICE: str = '저가'
    LST_PRC: str = '직전가격'
    ACC_TRDVOL: str = '누적체결수량'
    ACC_AMT: str = '누적거래대금'
    LST_ASKBID_TP_CD: str = '최종매도매수구분코드'
    LP_HD_QTY: str = 'LP보유수량'
    DATA_TYPE: str = '데이터구분'
    MSG_SEQ: str = '메세지일련번호'


@dataclass
class DerivativeTickFields:
    """선물/옵션 체결 필드 정의 (21개 필드 + 추가 4개)"""
    TRADE_DATE: str = '거래일자'
    REGUL_OFFHR_TP_CD: str = '정규시간외구분코드'
    ISIN_CODE: str = '종목코드'
    JONG_INDEX: str = '종목인덱스'
    TRD_PRC: str = '체결가격'
    TRDVOL: str = '체결수량'
    TRD_TP_CD: str = '체결유형코드'
    TRD_DD: str = '체결일자'
    TRD_TM: str = '체결시각'
    NBMM_TRD_PRC: str = '근월물체결가격'
    FUTRMM_TRD_PRC: str = '원월물체결가격'
    OPEN_PRICE: str = '시가'
    HIGH_PRICE: str = '고가'
    LOW_PRICE: str = '저가'
    LST_PRC: str = '직전가격'
    ACC_TRDVOL: str = '누적체결수량'
    ACC_AMT: str = '누적거래대금'
    LST_ASKBID_TP_CD: str = '최종매도매수구분코드'
    LP_HD_QTY: str = 'LP보유수량'
    DATA_TYPE: str = '데이터구분'
    MSG_SEQ: str = '메세지일련번호'
    # 2014.03 추가
    BRD_ID: str = '보드ID'
    SESSION_ID: str = '세션ID'
    # 2014.09 추가
    DYNMC_UPLMTPRC: str = '실시간상한가'
    DYNMC_LWLMTPRC: str = '실시간하한가'


class KoscomTickParser:
    """코스콤 틱데이터 통합 파서"""
    
    # 주식 필드명 (45개)
    STOCK_COLUMNS = [
        'TRADE_DATE', 'BLKTRD_TP_CD', 'REGUL_OFFHR_TP_CD', 'ISIN_CODE', 
        'JONG_INDEX', 'TRD_NO', 'TRD_PRC', 'TRDVOL', 'TRD_TP_CD', 'TRD_DD', 
        'TRD_TM', 'NBMM_TRD_PRC', 'FUTRMM_TRD_PRC', 'BID_MBR_NO', 'BIDORD_TP_CD',
        'BID_TRSTK_STAT_ID', 'BID_TRSTK_TRD_METHD_CD', 'BID_ASK_TP_CD', 
        'BID_TRST_PRINC_TP_CD', 'BID_TRSTCOM_NO', 'BID_PT_TP_CD', 'BID_INVST_TP_CD',
        'BID_FORNINVST_TP_CD', 'BIDORD_ACPT_NO', 'ASK_MBR_NO', 'ASKORD_TP_CD',
        'ASK_TRSTK_STAT_ID', 'ASK_TRSTK_TRD_METHD_CD', 'ASK_ASK_TP_CD',
        'ASK_TRST_PRINC_TP_CD', 'ASK_TRSTCOM_NO', 'ASK_PT_TP_CD', 'ASK_INVST_TP_CD',
        'ASK_FORNINVST_TP_CD', 'ASKORD_ACPT_NO', 'OPEN_PRICE', 'HIGH_PRICE',
        'LOW_PRICE', 'LST_PRC', 'ACC_TRDVOL', 'ACC_AMT', 'LST_ASKBID_TP_CD',
        'LP_HD_QTY', 'DATA_TYPE', 'MSG_SEQ'
    ]
    
    # 파생상품 필드명 (21개 기본)
    DERIVATIVE_COLUMNS_21 = [
        'TRADE_DATE', 'REGUL_OFFHR_TP_CD', 'ISIN_CODE', 'JONG_INDEX',
        'TRD_PRC', 'TRDVOL', 'TRD_TP_CD', 'TRD_DD', 'TRD_TM',
        'NBMM_TRD_PRC', 'FUTRMM_TRD_PRC', 'OPEN_PRICE', 'HIGH_PRICE',
        'LOW_PRICE', 'LST_PRC', 'ACC_TRDVOL', 'ACC_AMT', 'LST_ASKBID_TP_CD',
        'LP_HD_QTY', 'DATA_TYPE', 'MSG_SEQ'
    ]
    
    # 파생상품 필드명 (25개 - 2014년 이후)
    DERIVATIVE_COLUMNS_25 = DERIVATIVE_COLUMNS_21 + [
        'BRD_ID', 'SESSION_ID', 'DYNMC_UPLMTPRC', 'DYNMC_LWLMTPRC'
    ]
    
    @staticmethod
    def parse_stock_tick(gz_path: str, num_rows: Optional[int] = None) -> pd.DataFrame:
        """
        유가증권/코스닥 체결 틱데이터 파싱
        
        Args:
            gz_path: .dat.gz 파일 경로
            num_rows: 읽을 행 수 (None이면 전체)
        
        Returns:
            DataFrame
        """
        data = []
        
        with gzip.open(gz_path, 'rb') as f:
            for i, line in enumerate(f):
                if num_rows and i >= num_rows:
                    break
                    
                try:
                    decoded = line.decode('euc-kr').strip()
                except:
                    decoded = line.decode('cp949').strip()
                
                fields = decoded.split('|')
                data.append(fields)
        
        df = pd.DataFrame(data, columns=KoscomTickParser.STOCK_COLUMNS)
        
        # 데이터 타입 변환
        df['TRD_PRC'] = pd.to_numeric(df['TRD_PRC'], errors='coerce')
        df['TRDVOL'] = pd.to_numeric(df['TRDVOL'], errors='coerce')
        df['OPEN_PRICE'] = pd.to_numeric(df['OPEN_PRICE'], errors='coerce')
        df['HIGH_PRICE'] = pd.to_numeric(df['HIGH_PRICE'], errors='coerce')
        df['LOW_PRICE'] = pd.to_numeric(df['LOW_PRICE'], errors='coerce')
        df['ACC_TRDVOL'] = pd.to_numeric(df['ACC_TRDVOL'], errors='coerce')
        df['ACC_AMT'] = pd.to_numeric(df['ACC_AMT'], errors='coerce')
        
        return df
    
    @staticmethod
    def parse_derivative_tick(gz_path: str, num_rows: Optional[int] = None) -> pd.DataFrame:
        """
        선물/옵션 체결 틱데이터 파싱
        
        Args:
            gz_path: .dat.gz 파일 경로
            num_rows: 읽을 행 수 (None이면 전체)
        
        Returns:
            DataFrame
        """
        data = []
        field_count = None
        
        with gzip.open(gz_path, 'rb') as f:
            for i, line in enumerate(f):
                if num_rows and i >= num_rows:
                    break
                    
                try:
                    decoded = line.decode('euc-kr').strip()
                except:
                    decoded = line.decode('cp949').strip()
                
                fields = decoded.split('|')
                
                # 첫 줄에서 필드 개수 확인
                if field_count is None:
                    field_count = len(fields)
                
                data.append(fields)
        
        # 필드 개수에 따라 컬럼명 선택
        if field_count == 21:
            columns = KoscomTickParser.DERIVATIVE_COLUMNS_21
        elif field_count == 25:
            columns = KoscomTickParser.DERIVATIVE_COLUMNS_25
        else:
            print(f"경고: 예상치 못한 필드 개수 {field_count}개")
            columns = [f'FIELD_{i}' for i in range(field_count)]
        
        df = pd.DataFrame(data, columns=columns)
        
        # 데이터 타입 변환
        df['TRD_PRC'] = pd.to_numeric(df['TRD_PRC'], errors='coerce')
        df['TRDVOL'] = pd.to_numeric(df['TRDVOL'], errors='coerce')
        df['OPEN_PRICE'] = pd.to_numeric(df['OPEN_PRICE'], errors='coerce')
        df['HIGH_PRICE'] = pd.to_numeric(df['HIGH_PRICE'], errors='coerce')
        df['LOW_PRICE'] = pd.to_numeric(df['LOW_PRICE'], errors='coerce')
        df['ACC_TRDVOL'] = pd.to_numeric(df['ACC_TRDVOL'], errors='coerce')
        df['ACC_AMT'] = pd.to_numeric(df['ACC_AMT'], errors='coerce')
        
        # 옵션 가격 처리 (소수부만 표시된 경우)
        if df['ISIN_CODE'].str.startswith('KR42').any():
            print("옵션 데이터 감지: 가격은 소수부만 표시될 수 있습니다 (행사가 + 가격)")
        
        if field_count == 25:
            df['DYNMC_UPLMTPRC'] = pd.to_numeric(df['DYNMC_UPLMTPRC'], errors='coerce')
            df['DYNMC_LWLMTPRC'] = pd.to_numeric(df['DYNMC_LWLMTPRC'], errors='coerce')
        
        return df
    
    @staticmethod
    def auto_parse(gz_path: str, num_rows: Optional[int] = None) -> pd.DataFrame:
        """
        파일명 기반 자동 파싱
        
        Args:
            gz_path: .dat.gz 파일 경로
            num_rows: 읽을 행 수 (None이면 전체)
        
        Returns:
            DataFrame
        """
        filename = Path(gz_path).name.upper()
        
        # 파일명으로 상품 타입 판별
        if 'SKSN' in filename or 'SQSN' in filename:
            # 유가증권 또는 코스닥
            print(f"주식 틱데이터 파싱: {filename}")
            return KoscomTickParser.parse_stock_tick(gz_path, num_rows)
        elif 'DFKN' in filename or 'DOKN' in filename:
            # 선물 또는 옵션
            print(f"파생상품 틱데이터 파싱: {filename}")
            return KoscomTickParser.parse_derivative_tick(gz_path, num_rows)
        else:
            raise ValueError(f"알 수 없는 파일 형식: {filename}")


# 사용 예시
if __name__ == "__main__":
    
    # 예시 1: 선물 데이터 파싱
    futures_path = r"E:\선물 체결틱데이터(2010.Q1~2023.Q4)\DFKNXTRDSHRTH_2017_Q1.dat.gz"
    df_futures = KoscomTickParser.parse_derivative_tick(futures_path, num_rows=1000)
    print("\n선물 데이터:")
    print(df_futures.head())
    print(f"\n총 {len(df_futures):,}개 행")
    print(f"컬럼 수: {len(df_futures.columns)}개")
    
    # 예시 2: 옵션 데이터 파싱
    options_path = r"E:\옵션 체결틱데이터(2010.Q1~2023.Q4)\DOKNXTRDSHRTH_2012_Q1.dat.gz"
    df_options = KoscomTickParser.parse_derivative_tick(options_path, num_rows=1000)
    print("\n옵션 데이터:")
    print(df_options.head())
    
    # 예시 3: 유가증권 데이터 파싱
    stock_path = r"E:\유가증권 체결틱데이터(2010.1~2023.12)\SKSNXTRDIJH_2010_02.dat.txt.gz"
    df_stock = KoscomTickParser.parse_stock_tick(stock_path, num_rows=1000)
    print("\n유가증권 데이터:")
    print(df_stock.head())
    
    # 예시 4: 자동 파싱
    df_auto = KoscomTickParser.auto_parse(futures_path, num_rows=100)
    print("\n자동 파싱:")
    print(df_auto.head())
