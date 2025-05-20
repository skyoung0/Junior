/**
 * 데이터 서비스
 * 여행 플래너에 필요한 데이터를 로드하고 처리하는 서비스 함수들을 제공합니다.
 */

/**
 * 모든 여행지 정보 가져오기
 * @returns {Promise<Array>} 여행지 정보 배열
 */
export const getAllDestinations = async () => {
    try {
      const response = await fetch('/data/destinations.json');
      if (!response.ok) {
        throw new Error('목적지 데이터를 불러오는데 실패했습니다.');
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching destinations:', error);
      return [];
    }
  };
  
  /**
   * 특정 여행지 정보 가져오기
   * @param {string} destinationName - 여행지 이름
   * @returns {Promise<Object|null>} 여행지 정보 객체 또는 null
   */
  export const getDestinationByName = async (destinationName) => {
    try {
      const destinations = await getAllDestinations();
      return destinations.find(dest => dest.name === destinationName) || null;
    } catch (error) {
      console.error('Error fetching destination:', error);
      return null;
    }
  };
  
  /**
   * 특정 여행지의 기차 정보 가져오기
   * @param {string} destinationName - 여행지 이름
   * @returns {Promise<Object|null>} 기차 정보 객체 또는 null
   */
  export const getTrainInfoForDestination = async (destinationName) => {
    try {
      const response = await fetch('/data/trainTimes.json');
      if (!response.ok) {
        throw new Error('기차 데이터를 불러오는데 실패했습니다.');
      }
      const data = await response.json();
      return data.find(item => item.destination === destinationName) || null;
    } catch (error) {
      console.error('Error fetching train data:', error);
      return null;
    }
  };
  
  /**
   * 특정 여행지의 호텔 정보 가져오기
   * @param {string} destinationName - 여행지 이름
   * @returns {Promise<Array>} 호텔 정보 배열 또는 빈 배열
   */
  export const getHotelsForDestination = async (destinationName) => {
    try {
      const response = await fetch('/data/hotels.json');
      if (!response.ok) {
        throw new Error('호텔 데이터를 불러오는데 실패했습니다.');
      }
      const data = await response.json();
      const hotelData = data.find(item => item.destination === destinationName);
      return hotelData ? hotelData.hotels : [];
    } catch (error) {
      console.error('Error fetching hotel data:', error);
      return [];
    }
  };
  
  /**
   * 특정 여행지의 맛집 정보 가져오기
   * @param {string} destinationName - 여행지 이름
   * @returns {Promise<Array>} 맛집 정보 배열 또는 빈 배열
   */
  export const getRestaurantsForDestination = async (destinationName) => {
    try {
      const response = await fetch('/data/restaurants.json');
      if (!response.ok) {
        throw new Error('맛집 데이터를 불러오는데 실패했습니다.');
      }
      const data = await response.json();
      const restaurantData = data.find(item => item.destination === destinationName);
      return restaurantData ? restaurantData.restaurants : [];
    } catch (error) {
      console.error('Error fetching restaurant data:', error);
      return [];
    }
  };
  
  /**
   * AI 여행 일정 가져오기 (현재는 더미 데이터)
   * @param {string} destinationName - 여행지 이름
   * @returns {Promise<Object|null>} AI 여행 일정 또는 null
   */
  export const getAIItineraryForDestination = async (destinationName) => {
    try {
      // 실제 API 호출이 구현될 예정
      // 현재는 하드코딩된, 미리 정의된 일정 반환
      const dummyItineraries = {
        '서울': {
          title: '서울: 전통과 현대가 어우러진 3일 여행',
          description: '한국의 수도 서울에서 고궁부터 최신 트렌드 지역까지 다양한 매력을 경험하는 여행입니다.',
          days: [
            {
              day: 1,
              title: '서울의 역사 탐방',
              morning: {
                title: '경복궁 관람',
                description: '조선 왕조의 법궁인 경복궁을 둘러보세요. 근처의 국립고궁박물관도 함께 방문하면 좋습니다.'
              },
              afternoon: {
                title: '인사동 & 삼청동',
                description: '전통 공예품과 갤러리가 많은 인사동을, 이어서 카페와 부티크가 있는 삼청동을 거닐어보세요.'
              },
              evening: {
                title: '광화문 & 청계천',
                description: '저녁에는 야경이 아름다운 청계천을 산책하고, 광화문 주변의 맛집에서 저녁을 즐기세요.'
              }
            },
            {
              day: 2,
              title: '서울의 트렌디한 지역',
              morning: {
                title: '홍대 & 연남동',
                description: '젊음의 에너지가 넘치는 홍대 거리와 힙한 감성의 연남동을 방문해보세요.'
              },
              afternoon: {
                title: '여의도 한강공원',
                description: '여의도 한강공원에서 자전거를 빌려 한강변을 달리거나 피크닉을 즐겨보세요.'
              },
              evening: {
                title: '명동 & 동대문',
                description: '쇼핑의 메카 명동과 24시간 쇼핑이 가능한 동대문 디자인 플라자(DDP)를 방문해보세요.'
              }
            },
            {
              day: 3,
              title: '서울의 문화 체험',
              morning: {
                title: '남산 서울타워',
                description: '서울의 전망을 한눈에 볼 수 있는 남산 서울타워를 방문해보세요.'
              },
              afternoon: {
                title: '이태원 & 경리단길',
                description: '다양한 문화가 공존하는 이태원과 감각적인 카페와 레스토랑이 모여있는 경리단길을 탐험해보세요.'
              },
              evening: {
                title: '한강 유람선',
                description: '한강 유람선을 타고 야경을 감상하며 여행의 마무리를 장식해보세요.'
              }
            }
          ]
        },
        '부산': {
          title: '부산: 해변과 맛의 도시 3일 여행',
          description: '한국 제2의 도시 부산에서 아름다운 해변과 신선한 해산물, 그리고 독특한 문화를 경험하는 여행입니다.',
          days: [
            {
              day: 1,
              title: '부산의 대표 해변',
              morning: {
                title: '해운대 해변',
                description: '부산의 상징인 해운대 해변을 방문하여 여유로운 시간을 보내세요.'
              },
              afternoon: {
                title: '동백섬 & 누리마루',
                description: '해운대 근처 동백섬을 산책하고 APEC 정상회담이 열렸던 누리마루를 방문해보세요.'
              },
              evening: {
                title: '해운대 시장',
                description: '저녁에는 해운대 시장에서 다양한 해산물 요리를 맛보세요.'
              }
            },
            // 나머지 일정은 생략...
          ]
        },
        // 다른 도시들의 일정도 포함...
      };
      
      return dummyItineraries[destinationName] || null;
    } catch (error) {
      console.error('Error fetching AI itinerary:', error);
      return null;
    }
  };
  
  /**
   * 로그인 함수 (실제 인증 없음)
   * @param {string} username - 사용자 이름
   * @param {string} password - 비밀번호
   * @returns {Promise<{success: boolean, user?: object, message?: string}>} 로그인 결과
   */
  export const login = async (username, password) => {
    // 실제 API 인증을 시뮬레이션
    return new Promise(resolve => {
      setTimeout(() => {
        if (username === 'iksang' && password === '1234') {
          resolve({
            success: true,
            user: {
              username: 'iksang',
              name: '익상님',
              preferences: {
                favoriteDestinations: ['제주도', '부산']
              }
            }
          });
        } else {
          resolve({
            success: false,
            message: '아이디 또는 비밀번호가 올바르지 않습니다.'
          });
        }
      }, 500); // 실제 API 호출을 시뮬레이션하기 위한 지연
    });
  };
  
  /**
   * 여행지 검색 함수
   * @param {string} keyword - 검색어
   * @returns {Promise<Array>} 검색 결과 배열
   */
  export const searchDestinations = async (keyword) => {
    try {
      const destinations = await getAllDestinations();
      if (!keyword) return destinations;
      
      const lowercasedKeyword = keyword.toLowerCase();
      return destinations.filter(dest => 
        dest.name.toLowerCase().includes(lowercasedKeyword) ||
        dest.description.toLowerCase().includes(lowercasedKeyword) ||
        dest.tags.some(tag => tag.toLowerCase().includes(lowercasedKeyword))
      );
    } catch (error) {
      console.error('Error searching destinations:', error);
      return [];
    }
  };