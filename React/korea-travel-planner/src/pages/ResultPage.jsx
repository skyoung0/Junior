import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Train, Hotel, Utensils, Map, Calendar, Clock, ExternalLink } from 'lucide-react';

const ResultPage = () => {
  const { destination } = useParams();
  const [loading, setLoading] = useState(true);
  const [destinationData, setDestinationData] = useState(null);
  const [trainData, setTrainData] = useState([]);
  const [hotelData, setHotelData] = useState([]);
  const [restaurantData, setRestaurantData] = useState([]);
  const [departureCity, setDepartureCity] = useState('서울');
  const [tab, setTab] = useState('itinerary');

  // 더미 AI 추천 일정 (실제로는 API 호출로 대체될 부분)
  const dummyItinerary = {
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
        {
          day: 2,
          title: '부산의 문화와 역사',
          morning: {
            title: '감천문화마을',
            description: '알록달록한 집들이 모여 있는 감천문화마을에서 부산의 예술적 감성을 느껴보세요.'
          },
          afternoon: {
            title: '용두산공원 & 국제시장',
            description: '용두산공원의 부산타워에서 도시 전경을 감상하고, 이어서 국제시장에서 쇼핑과 먹거리를 즐기세요.'
          },
          evening: {
            title: '자갈치시장',
            description: '부산의 대표적인 수산시장인 자갈치시장에서 신선한 해산물 요리를 맛보세요.'
          }
        },
        {
          day: 3,
          title: '부산의 자연과 명소',
          morning: {
            title: '광안리 해변',
            description: '광안대교의 웅장한 모습을 볼 수 있는 광안리 해변을 방문해보세요.'
          },
          afternoon: {
            title: '태종대',
            description: '부산의 끝자락에 위치한 태종대에서 절벽과 바다의 장관을 감상하세요.'
          },
          evening: {
            title: '남포동 & 광복동',
            description: '쇼핑과 맛집이 가득한 남포동과 광복동에서 여행의 마무리를 장식하세요.'
          }
        }
      ]
    },
    '제주도': {
      title: '제주도: 아름다운 자연경관의 3일 여행',
      description: '화산섬 제주도에서 독특한 자연경관과 문화를 경험하는 여행입니다.',
      days: [
        {
          day: 1,
          title: '제주 동부 탐방',
          morning: {
            title: '성산일출봉',
            description: '유네스코 세계자연유산으로 등재된 성산일출봉에 오르며 여행을 시작해보세요.'
          },
          afternoon: {
            title: '우도',
            description: '성산항에서 배를 타고 우도로 건너가 아름다운 해변과 경관을 즐기세요.'
          },
          evening: {
            title: '함덕 해변',
            description: '하얀 모래와 에메랄드빛 바다가 인상적인 함덕 해변에서 석양을 감상하세요.'
          }
        },
        {
          day: 2,
          title: '제주 남부 탐방',
          morning: {
            title: '주상절리대',
            description: '제주의 독특한 지질 명소인 주상절리대를 방문해보세요.'
          },
          afternoon: {
            title: '카멜리아 힐',
            description: '아름다운 동백꽃 정원인 카멜리아 힐에서 산책을 즐기세요.'
          },
          evening: {
            title: '서귀포 맛집 탐방',
            description: '서귀포 지역의 맛집에서 제주의 특색 있는 음식을 즐기세요.'
          }
        },
        {
          day: 3,
          title: '제주 서부 탐방',
          morning: {
            title: '한라산 트레킹',
            description: '제주의 상징인 한라산 국립공원에서 트레킹을 즐겨보세요.'
          },
          afternoon: {
            title: '협재 해변',
            description: '에메랄드 빛 바다와 하얀 모래가 인상적인 협재 해변을 방문하세요.'
          },
          evening: {
            title: '오설록 티 뮤지엄',
            description: '녹차 밭과 티 뮤지엄이 있는 오설록에서 제주의 차 문화를 경험하세요.'
          }
        }
      ]
    },
    '경주': {
      title: '경주: 천년 고도의 역사 여행 2일',
      description: '신라의 수도였던 경주에서 한국의 풍부한 역사와 문화유산을 경험하는 여행입니다.',
      days: [
        {
          day: 1,
          title: '경주 역사 탐방 1',
          morning: {
            title: '불국사 & 석굴암',
            description: '유네스코 세계문화유산인 불국사와 석굴암을 방문하여 신라 시대의 불교 예술을 감상하세요.'
          },
          afternoon: {
            title: '대릉원 & 천마총',
            description: '신라 왕들의 무덤인 대릉원과 천마도가 발견된 천마총을 탐험해보세요.'
          },
          evening: {
            title: '동궁과 월지(안압지)',
            description: '야경이 아름다운 동궁과 월지(안압지)에서 신라의 정취를 느껴보세요.'
          }
        },
        {
          day: 2,
          title: '경주 역사 탐방 2',
          morning: {
            title: '경주국립박물관',
            description: '신라의 문화재를 총망라한 경주국립박물관에서 역사 공부를 해보세요.'
          },
          afternoon: {
            title: '첨성대 & 계림',
            description: '동양에서 가장 오래된 천문대인 첨성대와 신라 시조의 탄생설화가 있는 계림을 방문하세요.'
          },
          evening: {
            title: '황리단길',
            description: '경주의 핫플레이스로 떠오른 황리단길에서 카페와 맛집을 탐방하세요.'
          }
        }
      ]
    },
    '전주': {
      title: '전주: 한옥과 맛의 도시 2일 여행',
      description: '한옥마을과 비빔밥으로 유명한 전주에서 전통문화와 맛있는 음식을 즐기는 여행입니다.',
      days: [
        {
          day: 1,
          title: '전주 한옥마을 탐방',
          morning: {
            title: '경기전',
            description: '조선의 태조 이성계의 어진을 모신 경기전을 방문해보세요.'
          },
          afternoon: {
            title: '한옥마을 구석구석',
            description: '700여 채의 한옥이 모여있는 전주 한옥마을을 천천히 둘러보세요.'
          },
          evening: {
            title: '전주 막걸리와 야식',
            description: '전주의 명물인 막걸리와 함께 다양한 안주를 즐기세요.'
          }
        },
        {
          day: 2,
          title: '전주 맛집 탐방',
          morning: {
            title: '전주 비빔밥',
            description: '전주의 대표 음식인 비빔밥을 아침으로 즐겨보세요.'
          },
          afternoon: {
            title: '남부시장 & 풍남문',
            description: '전통시장인 남부시장과 전주성의 남문인 풍남문을 방문해보세요.'
          },
          evening: {
            title: '한옥마을 야경',
            description: '조명이 켜진 한옥마을의 아름다운 야경을 감상하며 여행을 마무리하세요.'
          }
        }
      ]
    },
    '강릉': {
      title: '강릉: 바다와 커피의 도시 2일 여행',
      description: '동해안의 아름다운 해변과 커피 문화가 발달한 강릉을 즐기는 여행입니다.',
      days: [
        {
          day: 1,
          title: '강릉 해변 탐방',
          morning: {
            title: '정동진 해변',
            description: '해돋이 명소로 유명한 정동진 해변에서 아름다운 동해의 풍경을 감상하세요.'
          },
          afternoon: {
            title: '경포대 & 경포호',
            description: '경포대와 주변의 아름다운 경포호 산책로를 걸어보세요.'
          },
          evening: {
            title: '안목 커피거리',
            description: '바다를 보며 커피를 즐길 수 있는 안목 커피거리에서 여유를 가지세요.'
          }
        },
        {
          day: 2,
          title: '강릉 문화 탐방',
          morning: {
            title: '오죽헌 & 선교장',
            description: '율곡 이이 선생의 생가인 오죽헌과 조선시대 대표적인 사대부가의 살림집인 선교장을 방문하세요.'
          },
          afternoon: {
            title: '주문진 수산시장',
            description: '신선한 해산물이 가득한 주문진 수산시장에서 맛있는 해산물을 맛보세요.'
          },
          evening: {
            title: '하슬라 아트월드',
            description: '예술 작품이 전시된 하슬라 아트월드를 방문하여 문화예술을 감상하세요.'
          }
        }
      ]
    }
  };

  // 데이터 로드
  useEffect(() => {
    const fetchData = async () => {
      try {
        // 목적지 정보 가져오기
        const destResponse = await fetch('/data/destinations.json');
        const destData = await destResponse.json();
        const selectedDest = destData.find(d => d.name === destination);
        setDestinationData(selectedDest);

        // 기차 정보 가져오기
        const trainResponse = await fetch('/data/trainTimes.json');
        const trainData = await trainResponse.json();
        const selectedTrainData = trainData.find(t => t.destination === destination);
        setTrainData(selectedTrainData?.routes || []);

        // 호텔 정보 가져오기
        const hotelResponse = await fetch('/data/hotels.json');
        const hotelData = await hotelResponse.json();
        const selectedHotelData = hotelData.find(h => h.destination === destination);
        setHotelData(selectedHotelData?.hotels || []);

        // 레스토랑 정보 가져오기
        const restaurantResponse = await fetch('/data/restaurants.json');
        const restaurantData = await restaurantResponse.json();
        const selectedRestaurantData = restaurantData.find(r => r.destination === destination);
        setRestaurantData(selectedRestaurantData?.restaurants || []);

        setLoading(false);
      } catch (error) {
        console.error('Error fetching data:', error);
        setLoading(false);
      }
    };

    fetchData();
  }, [destination]);

  return (
    <div className="bg-gray-50 min-h-screen">
      {loading ? (
        <div className="max-w-7xl mx-auto px-4 py-16 sm:px-6 lg:px-8 text-center">
          <p className="text-lg text-gray-600">데이터를 불러오는 중...</p>
        </div>
      ) : (
        <>
          {/* 헤더 배너 */}
          <div className="relative">
            <div className="absolute inset-0 bg-black">
              <img 
                src={destinationData?.image} 
                alt={destination} 
                className="w-full h-full object-cover opacity-60"
              />
            </div>
            <div className="relative max-w-7xl mx-auto px-4 py-24 sm:px-6 lg:px-8 text-center">
              <h1 className="text-4xl font-extrabold text-white sm:text-5xl">
                {destination} 여행 플랜
              </h1>
              <p className="mt-6 max-w-3xl mx-auto text-xl text-gray-300">
                {destinationData?.description || '완벽한 여행 계획을 AI가 제안합니다.'}
              </p>
            </div>
          </div>

          {/* 탭 메뉴 */}
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="border-b border-gray-200">
              <nav className="-mb-px flex">
                <button
                  onClick={() => setTab('itinerary')}
                  className={`${
                    tab === 'itinerary'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } whitespace-nowrap py-4 px-4 border-b-2 font-medium text-sm md:text-base flex items-center`}
                >
                  <Calendar className="h-5 w-5 md:mr-2" />
                  <span className="hidden md:inline">AI 추천 일정</span>
                </button>
                <button
                  onClick={() => setTab('train')}
                  className={`${
                    tab === 'train'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } whitespace-nowrap py-4 px-4 border-b-2 font-medium text-sm md:text-base flex items-center`}
                >
                  <Train className="h-5 w-5 md:mr-2" />
                  <span className="hidden md:inline">기차 시간</span>
                </button>
                <button
                  onClick={() => setTab('hotel')}
                  className={`${
                    tab === 'hotel'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } whitespace-nowrap py-4 px-4 border-b-2 font-medium text-sm md:text-base flex items-center`}
                >
                  <Hotel className="h-5 w-5 md:mr-2" />
                  <span className="hidden md:inline">숙소 정보</span>
                </button>
                <button
                  onClick={() => setTab('restaurant')}
                  className={`${
                    tab === 'restaurant'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } whitespace-nowrap py-4 px-4 border-b-2 font-medium text-sm md:text-base flex items-center`}
                >
                  <Utensils className="h-5 w-5 md:mr-2" />
                  <span className="hidden md:inline">맛집 정보</span>
                </button>
              </nav>
            </div>
          </div>

          {/* 콘텐츠 영역 */}
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {/* AI 추천 일정 탭 */}
            {tab === 'itinerary' && (
              <div>
                <div className="mb-8">
                  <h2 className="text-2xl font-bold text-gray-900">{dummyItinerary[destination]?.title || `${destination} 여행 계획`}</h2>
                  <p className="mt-2 text-gray-600">{dummyItinerary[destination]?.description || '완벽한 여행 계획을 AI가 추천해드립니다.'}</p>
                  <p className="mt-4 text-sm text-gray-500 bg-blue-50 p-3 rounded-md">
                    <span className="font-medium">AI 추천 사항:</span> 이 일정은 AI가 추천한 예시 일정입니다. 실제 GPT API 연동 시 더 개인화된 일정을 받아보실 수 있습니다.
                  </p>
                </div>

                {dummyItinerary[destination]?.days.map((day) => (
                  <div key={day.day} className="mb-8 border rounded-lg overflow-hidden shadow-sm">
                    <div className="px-6 py-4 bg-blue-600 text-white">
                      <h3 className="text-lg font-medium">Day {day.day}: {day.title}</h3>
                    </div>
                    <div className="p-6 grid grid-cols-1 md:grid-cols-3 gap-6">
                      {/* 오전 */}
                      <div className="bg-white p-4 rounded-lg shadow">
                        <div className="flex items-center mb-3">
                          <div className="bg-amber-100 rounded-full p-2 mr-3">
                            <Clock className="h-6 w-6 text-amber-600" />
                          </div>
                          <h4 className="text-lg font-medium text-gray-900">오전</h4>
                        </div>
                        <h5 className="font-medium text-gray-900">{day.morning.title}</h5>
                        <p className="mt-2 text-gray-600 text-sm">{day.morning.description}</p>
                      </div>

                      {/* 오후 */}
                      <div className="bg-white p-4 rounded-lg shadow">
                        <div className="flex items-center mb-3">
                          <div className="bg-blue-100 rounded-full p-2 mr-3">
                            <Clock className="h-6 w-6 text-blue-600" />
                          </div>
                          <h4 className="text-lg font-medium text-gray-900">오후</h4>
                        </div>
                        <h5 className="font-medium text-gray-900">{day.afternoon.title}</h5>
                        <p className="mt-2 text-gray-600 text-sm">{day.afternoon.description}</p>
                      </div>

                      {/* 저녁 */}
                      <div className="bg-white p-4 rounded-lg shadow">
                        <div className="flex items-center mb-3">
                          <div className="bg-indigo-100 rounded-full p-2 mr-3">
                            <Clock className="h-6 w-6 text-indigo-600" />
                          </div>
                          <h4 className="text-lg font-medium text-gray-900">저녁</h4>
                        </div>
                        <h5 className="font-medium text-gray-900">{day.evening.title}</h5>
                        <p className="mt-2 text-gray-600 text-sm">{day.evening.description}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* 기차 시간 탭 */}
            {tab === 'train' && (
              <div>
                <div className="mb-8">
                  <h2 className="text-2xl font-bold text-gray-900">{destination}행 기차</h2>
                  <p className="mt-2 text-gray-600">편리한 기차 시간을 확인하고 예약하세요.</p>
                </div>

                {destination === '제주도' ? (
                  <div className="bg-yellow-50 p-6 rounded-lg">
                    <p className="text-lg text-yellow-700">제주도는 기차로 갈 수 없는 섬입니다. 항공편을 이용해주세요.</p>
                    <a 
                      href="https://www.koreanair.com" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="mt-4 inline-flex items-center text-blue-600 hover:text-blue-800"
                    >
                      항공권 예약하기 <ExternalLink className="ml-1 h-4 w-4" />
                    </a>
                  </div>
                ) : (
                  <>
                    <div className="mb-6">
                      <label htmlFor="departureCity" className="block text-sm font-medium text-gray-700">출발지 선택</label>
                      <select
                        id="departureCity"
                        name="departureCity"
                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                        value={departureCity}
                        onChange={(e) => setDepartureCity(e.target.value)}
                      >
                        {trainData.map((route) => (
                          <option key={route.from} value={route.from}>{route.from}</option>
                        ))}
                      </select>
                    </div>

                    {trainData.length > 0 ? (
                      <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 md:rounded-lg">
                        <table className="min-w-full divide-y divide-gray-300">
                          <thead className="bg-gray-50">
                            <tr>
                              <th scope="col" className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 sm:pl-6">열차 종류</th>
                              <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">출발 시간</th>
                              <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">소요 시간</th>
                              <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">가격</th>
                              <th scope="col" className="relative py-3.5 pl-3 pr-4 sm:pr-6">
                                <span className="sr-only">예약</span>
                              </th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-200 bg-white">
                            {trainData.find(route => route.from === departureCity)?.departures.map((train, idx) => (
                              <tr key={idx}>
                                <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6">
                                  {trainData.find(route => route.from === departureCity)?.trainType}
                                </td>
                                <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{train.time}</td>
                                <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{train.duration}</td>
                                <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{train.price.toLocaleString()}원</td>
<td className="relative whitespace-nowrap py-4 pl-3 pr-4 text-right text-sm font-medium sm:pr-6">
 <a 
   href="https://www.letskorail.com/" 
   target="_blank" 
   rel="noopener noreferrer"
   className="text-blue-600 hover:text-blue-900 flex items-center justify-end"
 >
   예약하기 <ExternalLink className="ml-1 h-4 w-4" />
 </a>
</td>
                             </tr>
                           ))}
                         </tbody>
                       </table>
                     </div>
                   ) : (
                     <div className="bg-gray-50 p-6 rounded-lg">
                       <p className="text-gray-600">선택한 출발지에서 {destination}으로 가는 기차 정보가 없습니다.</p>
                     </div>
                   )}
                   
                   <div className="mt-6">
                     <a 
                       href="https://www.letskorail.com/" 
                       target="_blank" 
                       rel="noopener noreferrer"
                       className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                     >
                       코레일 홈페이지 방문하기 <ExternalLink className="ml-1 h-4 w-4" />
                     </a>
                   </div>
                 </>
               )}
             </div>
           )}

           {/* 숙소 정보 탭 */}
           {tab === 'hotel' && (
             <div>
               <div className="mb-8">
                 <h2 className="text-2xl font-bold text-gray-900">{destination} 추천 숙소</h2>
                 <p className="mt-2 text-gray-600">편안한 휴식을 위한 최적의 숙소를 선택하세요.</p>
               </div>

               <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
                 {hotelData.length > 0 ? (
                   hotelData.map((hotel) => (
                     <div key={hotel.id} className="bg-white overflow-hidden shadow-lg rounded-lg flex flex-col">
                       <div className="flex-shrink-0">
                         <img 
                           src={hotel.image} 
                           alt={hotel.name} 
                           className="h-48 w-full object-cover"
                         />
                       </div>
                       <div className="flex-1 p-6 flex flex-col justify-between">
                         <div className="flex-1">
                           <h3 className="text-xl font-semibold text-gray-900">{hotel.name}</h3>
                           <p className="mt-1 text-sm text-gray-500">{hotel.location}</p>
                           <div className="mt-2 flex items-center">
                             {[...Array(5)].map((_, i) => (
                               <svg 
                                 key={i} 
                                 className={`h-5 w-5 ${i < Math.floor(hotel.rating) ? 'text-yellow-400' : 'text-gray-300'}`}
                                 xmlns="http://www.w3.org/2000/svg" 
                                 viewBox="0 0 20 20" 
                                 fill="currentColor"
                               >
                                 <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                               </svg>
                             ))}
                             <span className="ml-2 text-sm text-gray-600">{hotel.rating}/5</span>
                           </div>
                           <p className="mt-3 text-gray-600">{hotel.description}</p>
                           <div className="mt-3">
                             <span className="text-gray-900 font-medium">1박 {hotel.pricePerNight.toLocaleString()}원</span>
                           </div>
                           <div className="mt-4 flex flex-wrap gap-2">
                             {hotel.amenities.map((amenity, idx) => (
                               <span 
                                 key={idx} 
                                 className="inline-block bg-gray-100 px-2 py-1 text-xs font-medium text-gray-600 rounded"
                               >
                                 {amenity}
                               </span>
                             ))}
                           </div>
                         </div>
                         <div className="mt-6">
                           <a 
                             href="https://www.goodchoice.kr/" 
                             target="_blank" 
                             rel="noopener noreferrer"
                             className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                           >
                             숙소 예약하기 <ExternalLink className="ml-1 h-4 w-4" />
                           </a>
                         </div>
                       </div>
                     </div>
                   ))
                 ) : (
                   <div className="col-span-full bg-gray-50 p-6 rounded-lg">
                     <p className="text-gray-600">{destination}의 숙소 정보를 찾을 수 없습니다.</p>
                   </div>
                 )}
               </div>
             </div>
           )}

           {/* 맛집 정보 탭 */}
           {tab === 'restaurant' && (
             <div>
               <div className="mb-8">
                 <h2 className="text-2xl font-bold text-gray-900">{destination} 추천 맛집</h2>
                 <p className="mt-2 text-gray-600">현지의 맛을, 현지의 분위기에서 즐겨보세요.</p>
               </div>

               <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
                 {restaurantData.length > 0 ? (
                   restaurantData.map((restaurant) => (
                     <div key={restaurant.id} className="bg-white overflow-hidden shadow-lg rounded-lg flex flex-col">
                       <div className="flex-shrink-0">
                         <img 
                           src={restaurant.image} 
                           alt={restaurant.name} 
                           className="h-48 w-full object-cover"
                         />
                       </div>
                       <div className="flex-1 p-6 flex flex-col justify-between">
                         <div className="flex-1">
                           <h3 className="text-xl font-semibold text-gray-900">{restaurant.name}</h3>
                           <div className="mt-1 flex items-center">
                             <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded">
                               {restaurant.cuisine}
                             </span>
                             <span className="ml-2 text-sm text-gray-500">{restaurant.location}</span>
                           </div>
                           <div className="mt-2 flex items-center">
                             {[...Array(5)].map((_, i) => (
                               <svg 
                                 key={i} 
                                 className={`h-5 w-5 ${i < Math.floor(restaurant.rating) ? 'text-yellow-400' : 'text-gray-300'}`}
                                 xmlns="http://www.w3.org/2000/svg" 
                                 viewBox="0 0 20 20" 
                                 fill="currentColor"
                               >
                                 <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                               </svg>
                             ))}
                             <span className="ml-2 text-sm text-gray-600">{restaurant.rating}/5</span>
                             <span className="ml-2 text-sm text-gray-500">가격대: {restaurant.priceRange}</span>
                           </div>
                           <p className="mt-3 text-gray-600">{restaurant.description}</p>
                           {restaurant.mustTry.length > 0 && (
                             <div className="mt-4">
                               <p className="text-sm font-medium text-gray-900">추천 메뉴:</p>
                               <ul className="mt-2 pl-5 text-sm text-gray-600 list-disc">
                                 {restaurant.mustTry.map((item, idx) => (
                                   <li key={idx}>{item}</li>
                                 ))}
                               </ul>
                             </div>
                           )}
                         </div>
                         <div className="mt-6">
                           <a 
                             href={`https://map.naver.com/v5/search/${encodeURIComponent(restaurant.name + ' ' + restaurant.location)}`} 
                             target="_blank" 
                             rel="noopener noreferrer"
                             className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                           >
                             지도에서 보기 <Map className="ml-1 h-4 w-4" />
                           </a>
                         </div>
                       </div>
                     </div>
                   ))
                 ) : (
                   <div className="col-span-full bg-gray-50 p-6 rounded-lg">
                     <p className="text-gray-600">{destination}의 맛집 정보를 찾을 수 없습니다.</p>
                   </div>
                 )}
               </div>
             </div>
           )}
         </div>
       </>
     )}
   </div>
 );
};

export default ResultPage;