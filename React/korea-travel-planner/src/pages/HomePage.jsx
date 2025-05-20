import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, MapPin, PlaneTakeoff, Calendar, Clock } from 'lucide-react';

const HomePage = () => {
  const [destinations, setDestinations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDestination, setSelectedDestination] = useState('');
  const navigate = useNavigate();

  // JSON 데이터 불러오기
  useEffect(() => {
    const fetchDestinations = async () => {
      try {
        const response = await fetch('/data/destinations.json');
        const data = await response.json();
        setDestinations(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching destinations:', error);
        setLoading(false);
      }
    };

    fetchDestinations();
  }, []);

  // 여행 계획 생성 핸들러
  const handleCreatePlan = (e) => {
    e.preventDefault();
    if (selectedDestination) {
      navigate(`/results/${selectedDestination}`);
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* 히어로 섹션 */}
      <div className="relative">
        {/* 배경 이미지 */}
        <div className="absolute inset-0 bg-black">
          <img 
            src="https://images.unsplash.com/photo-1538485399081-7a66562c35fc?q=80&w=1974&auto=format&fit=crop" 
            alt="서울 야경" 
            className="w-full h-full object-cover opacity-70"
          />
        </div>
        
        {/* 히어로 콘텐츠 */}
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 md:py-32">
          <div className="text-center">
            <h1 className="text-4xl md:text-5xl font-extrabold text-white tracking-tight sm:text-6xl">
              AI와 함께하는 한국 여행
            </h1>
            <p className="mt-6 max-w-2xl mx-auto text-xl text-gray-300">
              원하는 목적지만 선택하면 AI가 완벽한 여행 계획을 제안해 드립니다.
              기차 시간, 숙소, 맛집, 관광지까지 한번에!
            </p>
            
            {/* 검색 폼 */}
            <div className="mt-10 max-w-xl mx-auto">
              <form onSubmit={handleCreatePlan} className="bg-white p-4 rounded-lg shadow-xl md:flex">
                <div className="flex-grow mb-4 md:mb-0 md:mr-4">
                  <label htmlFor="destination" className="sr-only">목적지 선택</label>
                  <div className="relative rounded-md shadow-sm">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <MapPin className="h-5 w-5 text-gray-400" />
                    </div>
                    <select
                      id="destination"
                      name="destination"
                      className="block w-full pl-10 pr-12 py-3 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 rounded-md"
                      value={selectedDestination}
                      onChange={(e) => setSelectedDestination(e.target.value)}
                      required
                    >
                      <option value="">여행지 선택</option>
                      {!loading && destinations.map((dest) => (
                        <option key={dest.id} value={dest.name}>
                          {dest.name}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <button
                  type="submit"
                  className="w-full md:w-auto inline-flex justify-center items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                  disabled={!selectedDestination}
                >
                  <Search className="h-5 w-5 mr-2" />
                  여행 계획 생성
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>

      {/* 주요 기능 소개 */}
      <div className="py-12 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
              AI 여행 플래너의 특별한 기능
            </h2>
            <p className="mt-4 text-xl text-gray-600">
              복잡한 계획은 AI에게 맡기고, 여행의 즐거움에만 집중하세요
            </p>
          </div>

          <div className="mt-16">
            <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
              {/* 기능 1 */}
              <div className="flex flex-col items-center p-6 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-center h-14 w-14 rounded-full bg-blue-100 text-blue-600">
                  <PlaneTakeoff className="h-8 w-8" />
                </div>
                <h3 className="mt-4 text-lg font-medium text-gray-900">AI 맞춤 여행 계획</h3>
                <p className="mt-2 text-base text-gray-600 text-center">
                  선택한 목적지에 따라 AI가 최적의 여행 일정을 자동으로 생성합니다.
                </p>
              </div>

              {/* 기능 2 */}
              <div className="flex flex-col items-center p-6 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-center h-14 w-14 rounded-full bg-blue-100 text-blue-600">
                  <Calendar className="h-8 w-8" />
                </div>
                <h3 className="mt-4 text-lg font-medium text-gray-900">편리한 교통 정보</h3>
                <p className="mt-2 text-base text-gray-600 text-center">
                  기차 시간과 예약 링크를 제공하여 교통편 예약이 쉽고 빠릅니다.
                </p>
              </div>

              {/* 기능 3 */}
              <div className="flex flex-col items-center p-6 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-center h-14 w-14 rounded-full bg-blue-100 text-blue-600">
                  <Clock className="h-8 w-8" />
                </div>
                <h3 className="mt-4 text-lg font-medium text-gray-900">숙소 및 맛집 추천</h3>
                <p className="mt-2 text-base text-gray-600 text-center">
                  지역별 추천 숙소와 맛집 정보로 완벽한 여행 경험을 제공합니다.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* 추천 여행지 섹션 */}
      <div className="py-12 bg-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
              인기 여행지
            </h2>
            <p className="mt-4 max-w-2xl mx-auto text-xl text-gray-600">
              한국의 아름다운 여행지를 탐험해보세요
            </p>
          </div>

          {loading ? (
            <div className="mt-12 text-center">
              <p className="text-gray-600">여행지 정보를 불러오는 중...</p>
            </div>
          ) : (
            <div className="mt-12 grid gap-8 md:grid-cols-2 lg:grid-cols-3">
              {destinations.slice(0, 6).map((destination) => (
                <div key={destination.id} className="bg-white overflow-hidden shadow-lg rounded-lg">
                  <img 
                    src={destination.image} 
                    alt={destination.name} 
                    className="w-full h-48 object-cover"
                  />
                  <div className="p-6">
                    <h3 className="text-xl font-semibold text-gray-900">{destination.name}</h3>
                    <p className="mt-2 text-gray-600">{destination.shortDescription}</p>
                    <div className="mt-4 flex flex-wrap gap-2">
                      {destination.tags.map((tag, index) => (
                        <span 
                          key={index} 
                          className="inline-block bg-blue-100 text-blue-700 px-2 py-1 text-xs font-medium rounded"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                    <div className="mt-6">
                      <button
                        onClick={() => {
                          setSelectedDestination(destination.name);
                          navigate(`/results/${destination.name}`);
                        }}
                        className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                      >
                        여행 계획 만들기
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* CTA 섹션 */}
      <div className="bg-blue-700">
        <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:py-16 lg:px-8 lg:flex lg:items-center lg:justify-between">
          <h2 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            <span className="block">한국 여행을 계획 중이신가요?</span>
            <span className="block text-blue-200">지금 바로 AI의 도움을 받아보세요</span>
          </h2>
          <div className="mt-8 flex lg:mt-0 lg:flex-shrink-0">
            <div className="inline-flex rounded-md shadow">
              <a
                href="#"
                onClick={(e) => {
                  e.preventDefault();
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                className="inline-flex items-center justify-center px-5 py-3 border border-transparent text-base font-medium rounded-md text-blue-700 bg-white hover:bg-blue-50"
              >
                시작하기
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;