/**
 * 유틸리티 헬퍼 함수
 */

/**
 * 가격 포맷팅 함수
 * @param {number} price - 포맷팅할 가격
 * @returns {string} 포맷팅된 가격 문자열
 */
export const formatPrice = (price) => {
    return price.toLocaleString('ko-KR') + '원';
  };
  
  /**
   * 날짜 포맷팅 함수
   * @param {Date} date - 포맷팅할 날짜
   * @returns {string} YYYY-MM-DD 형식의 날짜 문자열
   */
  export const formatDate = (date) => {
    return date.toISOString().split('T')[0];
  };
  
  /**
   * 오늘 날짜를 기준으로 n일 후의 날짜 계산
   * @param {number} days - 추가할 일수
   * @returns {Date} 계산된 날짜
   */
  export const addDays = (days) => {
    const date = new Date();
    date.setDate(date.getDate() + days);
    return date;
  };
  
  /**
   * 현재 시간이 주간인지 야간인지 확인
   * @returns {boolean} 주간이면 true, 야간이면 false
   */
  export const isDaytime = () => {
    const hours = new Date().getHours();
    return hours >= 6 && hours < 18;
  };
  
  /**
   * 문자열 길이 제한 함수 (말줄임표 추가)
   * @param {string} str - 원본 문자열
   * @param {number} maxLength - 최대 길이
   * @returns {string} 제한된 문자열
   */
  export const truncateString = (str, maxLength) => {
    if (str.length <= maxLength) return str;
    return str.substring(0, maxLength) + '...';
  };
  
  /**
   * 로컬 스토리지에 데이터 저장
   * @param {string} key - 키
   * @param {any} value - 저장할 값
   */
  export const saveToLocalStorage = (key, value) => {
    try {
      const serializedValue = JSON.stringify(value);
      localStorage.setItem(key, serializedValue);
    } catch (error) {
      console.error('Error saving to localStorage:', error);
    }
  };
  
  /**
   * 로컬 스토리지에서 데이터 불러오기
   * @param {string} key - 키
   * @param {any} defaultValue - 기본값
   * @returns {any} 저장된 값 또는 기본값
   */
  export const loadFromLocalStorage = (key, defaultValue = null) => {
    try {
      const serializedValue = localStorage.getItem(key);
      if (serializedValue === null) return defaultValue;
      return JSON.parse(serializedValue);
    } catch (error) {
      console.error('Error loading from localStorage:', error);
      return defaultValue;
    }
  };
  
  /**
   * 여행지 데이터가 유효한지 확인하는 함수
   * @param {Object} destination - 여행지 데이터
   * @returns {boolean} 유효성 여부
   */
  export const isValidDestination = (destination) => {
    return (
      destination &&
      typeof destination === 'object' &&
      typeof destination.name === 'string' &&
      typeof destination.description === 'string' &&
      Array.isArray(destination.attractions)
    );
  };
  
  /**
   * 랜덤한 여행지 추천 함수
   * @param {Array} destinations - 여행지 배열
   * @param {number} count - 추천 개수
   * @returns {Array} 추천된 여행지 배열
   */
  export const getRandomDestinations = (destinations, count = 3) => {
    if (!Array.isArray(destinations) || destinations.length === 0) return [];
    
    const shuffled = [...destinations].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, Math.min(count, destinations.length));
  };