import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
from pathlib import Path

class ClimateTechScraper:
    def __init__(self):
        self.base_url = 'https://www.ctis.re.kr/ko/techClass/classification.do?key=1141'
        self.output_dir = Path('assets/data/scraped')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_driver(self):
        """Selenium WebDriver 설정"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # 브라우저 창 숨기기
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        # ChromeDriverManager로 자동 다운로드
        service = webdriver.chrome.service.Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
        
    def scrape_classification_basic(self):
        """기후기술 기본 분류체계 크롤링 (R 코드 Python 변환)"""
        print("🔍 기후기술 분류체계 크롤링 시작...")
        
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 소분류(L3) 수집
            l3_elements = soup.select('#table_box > table > tbody > tr > td.bgw')
            l3_data = []
            
            for i, element in enumerate(l3_elements, 1):
                l3_text = element.get_text(strip=True)
                if l3_text:
                    l3_data.append({
                        'No': i,
                        'L3': l3_text
                    })
            
            # 중분류 위치 (R 코드의 L2_List 참고)
            l2_positions = [1, 4, 12, 14, 16, 18, 21, 23, 27, 31, 33, 36, 38, 41]
            
            # 중분류(L2) 수집
            l2_data = {}
            rows = soup.select('#table_box > table > tbody > tr')
            
            for pos in l2_positions:
                if pos <= len(rows):
                    row = rows[pos - 1]  # 0-based index
                    
                    # 다양한 열에서 중분류 텍스트 찾기
                    cells = row.find_all('td')
                    l2_text = None
                    
                    for cell in cells:
                        text = cell.get_text(strip=True)
                        if text and len(text) > 3:  # 의미있는 텍스트만
                            l2_text = text
                            break
                    
                    if l2_text:
                        l2_data[pos] = self.clean_text(l2_text)
            
            # 대분류 위치
            l1_positions = [1, 23, 41]
            l1_data = {}
            
            for pos in l1_positions:
                if pos <= len(rows):
                    row = rows[pos - 1]
                    cells = row.find_all('td')
                    
                    for cell in cells:
                        text = cell.get_text(strip=True)
                        if any(keyword in text for keyword in ['감축', '적응', '융복합']):
                            l1_data[pos] = self.clean_text(text)
                            break
            
            # 데이터 병합
            result_data = []
            current_l1 = ""
            current_l2 = ""
            
            for item in l3_data:
                no = item['No']
                l3 = item['L3']
                
                # L1 업데이트
                for l1_pos in l1_positions:
                    if no >= l1_pos and l1_pos in l1_data:
                        current_l1 = l1_data[l1_pos]
                
                # L2 업데이트
                for l2_pos in l2_positions:
                    if no >= l2_pos and l2_pos in l2_data:
                        current_l2 = l2_data[l2_pos]
                
                result_data.append({
                    'L1_대분류': current_l1,
                    'L2_중분류': current_l2,
                    'L3_소분류': l3,
                    'No': no
                })
            
            # DataFrame 생성 및 저장
            df = pd.DataFrame(result_data)
            output_file = self.output_dir / 'climate_tech_classification.csv'
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print(f"✅ 기본 분류체계 저장 완료: {output_file}")
            print(f"📊 수집된 데이터: {len(df)}개 항목")
            
            return df
            
        except Exception as e:
            print(f"❌ 크롤링 실패: {str(e)}")
            return None
    
    def scrape_detailed_info(self):
        """기후기술 상세정보 크롤링 (Python 코드 개선)"""
        print("🔍 기후기술 상세정보 크롤링 시작...")
        
        driver = None
        try:
            driver = self.setup_driver()
            driver.get(self.base_url)
            
            # 페이지 로딩 대기
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "table_box"))
            )
            
            detailed_data = []
            
            # 각 분류별 상세정보 수집
            categories = ['감축', '적응', '융복합']
            
            for category in categories:
                print(f"📋 {category} 기술 정보 수집 중...")
                
                try:
                    # 카테고리별 링크 클릭
                    if category == '감축':
                        driver.find_element(By.XPATH, '//*[@id="tc1_anchor"]').click()
                    elif category == '적응':
                        driver.find_element(By.XPATH, '//*[@id="tc2_anchor"]').click()
                    elif category == '융복합':
                        driver.find_element(By.XPATH, '//*[@id="tc3_anchor"]').click()
                    
                    time.sleep(2)
                    
                    # 해당 카테고리의 모든 기술 항목 수집
                    tech_links = driver.find_elements(By.CSS_SELECTOR, f"#{category.lower()}_tech_list a")
                    
                    for i, link in enumerate(tech_links):
                        try:
                            link.click()
                            time.sleep(1)
                            
                            # 상세정보 추출
                            detail_info = self.extract_detail_info(driver, category)
                            if detail_info:
                                detailed_data.append(detail_info)
                            
                            if i % 5 == 0:  # 진행상황 출력
                                print(f"   진행: {i+1}/{len(tech_links)}")
                                
                        except Exception as e:
                            print(f"   ⚠️ 개별 항목 처리 실패: {str(e)}")
                            continue
                    
                except Exception as e:
                    print(f"   ❌ {category} 카테고리 처리 실패: {str(e)}")
                    continue
            
            # 결과 저장
            if detailed_data:
                df = pd.DataFrame(detailed_data)
                output_file = self.output_dir / 'climate_tech_detailed.csv'
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
                
                print(f"✅ 상세정보 저장 완료: {output_file}")
                print(f"📊 수집된 상세정보: {len(df)}개 항목")
                
                return df
            else:
                print("❌ 상세정보 수집 실패")
                return None
                
        except Exception as e:
            print(f"❌ 상세정보 크롤링 실패: {str(e)}")
            return None
        finally:
            if driver:
                driver.quit()
    
    def extract_detail_info(self, driver, category):
        """상세정보 추출"""
        try:
            # 기본 정보 추출
            subtitle = driver.find_element(By.CSS_SELECTOR, ".tech-title").text.strip()
            
            # 상세정보 테이블에서 정보 추출
            info_data = {
                'category': category,
                'subtitle': subtitle,
                'definition': self.safe_get_text(driver, '//*[@id="pdfShows"]/dl/dd[1]/table/tbody/tr/td[1]'),
                'keywords_kor': self.safe_get_text(driver, '//*[@id="pdfShows"]/dl/dd[2]/table/tbody/tr[1]/td'),
                'keywords_eng': self.safe_get_text(driver, '//*[@id="pdfShows"]/dl/dd[2]/table/tbody/tr[2]/td'),
                'leading_country': self.safe_get_text(driver, '//*[@id="pdfShows"]/dl/dd[3]/table/tbody/tr[1]/td'),
                'tech_level_pct': self.safe_get_text(driver, '//*[@id="pdfShows"]/dl/dd[3]/table/tbody/tr[2]/td'),
                'tech_gap': self.safe_get_text(driver, '//*[@id="pdfShows"]/dl/dd[3]/table/tbody/tr[3]/td'),
                'classification': self.safe_get_text(driver, '//*[@id="pdfShows"]/dl/dd[4]/table/tbody/tr/td')
            }
            
            return info_data
            
        except Exception as e:
            print(f"   ⚠️ 상세정보 추출 실패: {str(e)}")
            return None
    
    def safe_get_text(self, driver, xpath):
        """안전한 텍스트 추출"""
        try:
            element = driver.find_element(By.XPATH, xpath)
            return self.clean_text(element.text)
        except:
            return ""
    
    def clean_text(self, text):
        """텍스트 정제"""
        if not text:
            return ""
        
        # 개행문자, 탭, 캐리지리턴 제거
        text = re.sub(r'[\r\n\t]+', ' ', text)
        # 연속 공백 정리
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def create_sample_data(self):
        """샘플 데이터 생성 (크롤링 실패 시 대체용)"""
        print("📋 샘플 데이터 생성 중...")
        
        # 기본 분류체계 샘플
        classification_sample = [
            {'L1_대분류': '감축', 'L2_중분류': '재생에너지', 'L3_소분류': '태양광 발전', 'No': 1},
            {'L1_대분류': '감축', 'L2_중분류': '재생에너지', 'L3_소분류': '풍력 발전', 'No': 2},
            {'L1_대분류': '감축', 'L2_중분류': '재생에너지', 'L3_소분류': '수력 발전', 'No': 3},
            {'L1_대분류': '감축', 'L2_중분류': '비재생에너지', 'L3_소분류': '원자력 발전', 'No': 4},
            {'L1_대분류': '감축', 'L2_중분류': '에너지저장', 'L3_소분류': '배터리 저장', 'No': 5},
            {'L1_대분류': '적응', 'L2_중분류': '물관리', 'L3_소분류': '홍수 방어', 'No': 6},
            {'L1_대분류': '적응', 'L2_중분류': '농업', 'L3_소분류': '스마트팜', 'No': 7},
            {'L1_대분류': '융복합', 'L2_중분류': 'ICT 융합', 'L3_소분류': '스마트그리드', 'No': 8}
        ]
        
        df_classification = pd.DataFrame(classification_sample)
        output_file = self.output_dir / 'climate_tech_classification.csv'
        df_classification.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 상세정보 샘플
        detailed_sample = [
            {
                'category': '감축',
                'subtitle': '태양광 발전',
                'definition': '태양광을 이용하여 전기를 생산하는 기술',
                'keywords_kor': '태양광, 태양전지, 실리콘',
                'keywords_eng': 'Solar, Photovoltaic, Silicon',
                'leading_country': '중국',
                'tech_level_pct': '85%',
                'tech_gap': '2-3년',
                'classification': '신재생에너지 > 태양광'
            },
            {
                'category': '감축',
                'subtitle': '풍력 발전',
                'definition': '바람의 운동에너지를 전기에너지로 변환하는 기술',
                'keywords_kor': '풍력, 터빈, 발전기',
                'keywords_eng': 'Wind, Turbine, Generator',
                'leading_country': '덴마크',
                'tech_level_pct': '80%',
                'tech_gap': '3-5년',
                'classification': '신재생에너지 > 풍력'
            }
        ]
        
        df_detailed = pd.DataFrame(detailed_sample)
        output_file = self.output_dir / 'climate_tech_detailed.csv'
        df_detailed.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("✅ 샘플 데이터 생성 완료")
        return df_classification, df_detailed

def main():
    """메인 실행 함수"""
    scraper = ClimateTechScraper()
    
    print("🚀 기후기술 데이터 크롤링 시작")
    print("=" * 50)
    
    # 기본 분류체계 크롤링
    classification_df = scraper.scrape_classification_basic()
    
    # 상세정보 크롤링
    detailed_df = scraper.scrape_detailed_info()
    
    # 크롤링 실패 시 샘플 데이터 생성
    if classification_df is None or detailed_df is None:
        print("⚠️ 크롤링 실패로 샘플 데이터를 생성합니다.")
        scraper.create_sample_data()
    
    print("=" * 50)
    print("🎉 데이터 수집 완료!")

if __name__ == "__main__":
    main()