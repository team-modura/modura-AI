# utils/s3_service.py

import boto3
import os
from typing import List, Optional
from botocore.exceptions import ClientError
from botocore.config import Config

class S3Service:
    """
    AWS S3 서비스 유틸리티
    - Presigned URL 생성 (업로드/조회)
    - 파일 삭제
    """
    
    def __init__(self):
        """S3 클라이언트 초기화"""
        self.bucket = os.getenv("AWS_S3_BUCKET")
        self.region = os.getenv("AWS_REGION", "ap-northeast-2")
        
        if not self.bucket:
            raise ValueError("AWS_S3_BUCKET 환경변수가 설정되지 않았습니다.")
        
        # S3 클라이언트 설정
        self.s3_client = boto3.client(
            's3',
            region_name=self.region,
            config=Config(signature_version='s3v4')
        )
    
    def generate_view_presigned_url(self, key: str, expiration: int = 600) -> Optional[str]:
        """
        S3 파일 조회용 Presigned URL 생성 (GET)
        """
        if not key:
            return None
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket,
                    'Key': key
                },
                ExpiresIn=expiration
            )
            return url
        
        except ClientError as e:
            print(f"❌ Presigned URL 생성 실패: {e}")
            return None
    
    def generate_view_presigned_urls(self, keys: List[str], expiration: int = 600) -> List[str]:
        """
        여러 S3 파일의 Presigned URL을 일괄 생성
        """
        urls = []
        for key in keys:
            url = self.generate_view_presigned_url(key, expiration)
            if url:
                urls.append(url)
        return urls
    
    def extract_key_from_url(self, url: str) -> Optional[str]:
        """
        S3 URL에서 키 추출
        """
        try:
            # amazonaws.com/ 이후의 경로 추출
            if ".amazonaws.com/" in url:
                key = url.split(".amazonaws.com/", 1)[1]
                
                # 버킷 이름이 경로에 포함된 경우 제거
                if key.startswith(f"{self.bucket}/"):
                    key = key[len(self.bucket) + 1:]
                
                # 쿼리 파라미터 제거 (Presigned URL의 경우)
                if "?" in key:
                    key = key.split("?")[0]
                
                return key
            
            return None
        
        except Exception as e:
            print(f"URL에서 키 추출 실패: {e}")
            return None
    
    def generate_thumbnail_url(self, s3_key: Optional[str], expiration: int = 600) -> Optional[str]:
        """
        썸네일 URL 생성 (키가 있으면 Presigned URL 반환)
        """
        if s3_key and s3_key.strip():
            return self.generate_view_presigned_url(s3_key, expiration)
        return None


s3_service = S3Service()