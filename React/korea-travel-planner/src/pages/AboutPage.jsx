import React from 'react';
import { Link } from 'react-router-dom';
import { MapPin, PlaneTakeoff, Train, Hotel, Utensils, Calendar } from 'lucide-react';

const AboutPage = () => {
  return (
    <div className="bg-white">
      {/* 헤더 배너 */}
      <div className="relative">
        <div className="absolute inset-0 bg-black">
          <img 
            src="https://images.unsplash.com/photo-1540469515203-85a5289f9fab?q=80&w=1974&auto=format&fit=crop" 
            alt="경주 불국사" 
            className="w-full h-full object-cover opacity-60"
          />
        </div>
        <div className="relative max-w-7xl mx-auto px-4 py-24 sm:px-6 lg:px-8 text-center">
          <h1 className="text-4xl font-extrabold text-white sm:text-5xl">
            한국 AI 여행 플래너 소개
          </h1>
          <p className="mt-6 max-w-3xl mx-auto text-xl text-gray-300">
            AI의 도움으로 더 쉽고 즐거운 한국 여행을 계획하세요
          </p>
        </div>
      </div>

      {/* 소개 섹션 */}
      <div className="py-16 overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-base font-semibold text-blue-600 tracking-wide uppercase">About Us</h2>
            <p className="mt-1 text-3xl font-extrabold text-gray-900 sm:text-4xl">
              한국 여행의 새로운 경험
            </p>
            <p className="max-w-xl mt-5 mx-auto text-xl text-gray-600">
              한국 AI 여행 플래너는 AI 기술을 활용하여 개인 맞춤형 여행 계획을 제공하는 서비스입니다.
              복잡한 계획 과정을 단순화하여 더 편안하고 즐거운 여행을 준비할 수 있도록 도와드립니다.
            </p>
          </div>

          {/* 주요 기능 */}
          <div className="mt-16">
            <h3 className="text-2xl font-bold text-gray-900 text-center mb-8">
              주요 기능
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              <div className="bg-gray-50 p-6 rounded-lg shadow-sm">
                <div className="flex items-center mb-4">
                  <div className="flex-shrink-0 bg-blue-100 rounded-full p-3">
                    <Calendar className="h-6 w-6 text-blue-600" />
                  </div>
                  <h4 className="ml-4 text-lg font-medium text-gray-900">AI 맞춤 여행 계획</h4>
                </div>
                <p className="text-gray-600">
                  인공지능이 여행지의 특성, 계절, 관심사 등을 고려하여 최적의 여행 일정을 제안합니다.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg shadow-sm">
                <div className="flex items-center mb-4">
                  <div className="flex-shrink-0 bg-blue-100 rounded-full p-3">
                    <Train className="h-6 w-6 text-blue-600" />
                  </div>
                  <h4 className="ml-4 text-lg font-medium text-gray-900">기차 시간 정보</h4>
                </div>
                <p className="text-gray-600">
                  주요 도시 간 기차 시간과 가격 정보를 제공하고 예약 사이트로 연결해 드립니다.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg shadow-sm">
                <div className="flex items-center mb-4">
                  <div className="flex-shrink-0 bg-blue-100 rounded-full p-3">
                    <Hotel className="h-6 w-6 text-blue-600" />
                  </div>
                  <h4 className="ml-4 text-lg font-medium text-gray-900">숙소 추천</h4>
                </div>
                <p className="text-gray-600">
                  여행지별 추천 숙소 정보와 시설, 가격 등의 상세 정보를 제공합니다.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg shadow-sm">
                <div className="flex items-center mb-4">
                  <div className="flex-shrink-0 bg-blue-100 rounded-full p-3">
                    <Utensils className="h-6 w-6 text-blue-600" />
                  </div>
                  <h4 className="ml-4 text-lg font-medium text-gray-900">맛집 정보</h4>
                </div>
                <p className="text-gray-600">
                  지역별 유명 맛집과 추천 메뉴 정보를 제공하여 미식 여행을 도와드립니다.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg shadow-sm">
                <div className="flex items-center mb-4">
                  <div className="flex-shrink-0 bg-blue-100 rounded-full p-3">
                    <MapPin className="h-6 w-6 text-blue-600" />
                  </div>
                  <h4 className="ml-4 text-lg font-medium text-gray-900">지역 정보</h4>
                </div>
                <p className="text-gray-600">
                  주요 관광지, 볼거리, 계절별 특색 있는 행사 등 지역에 대한 유용한 정보를 제공합니다.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg shadow-sm">
                <div className="flex items-center mb-4">
                  <div className="flex-shrink-0 bg-blue-100 rounded-full p-3">
                    <PlaneTakeoff className="h-6 w-6 text-blue-600" />
                  </div>
                  <h4 className="ml-4 text-lg font-medium text-gray-900">외부 예약 연결</h4>
                </div>
                <p className="text-gray-600">
                  교통편과 숙소 예약을 위한 외부 사이트로의 연결을 제공합니다.
                </p>
              </div>
            </div>
          </div>

          {/* 기술 스택 */}
          <div className="mt-16">
            <h3 className="text-2xl font-bold text-gray-900 text-center mb-8">
              사용 기술
            </h3>
            <div className="bg-gray-50 rounded-lg p-8 shadow-sm">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-lg font-medium text-gray-900 mb-4">프론트엔드</h4>
                  <ul className="space-y-2 text-gray-600">
                    <li className="flex items-center">
                      <span className="inline-block bg-blue-100 rounded-full w-2 h-2 mr-2"></span>
                      React - 사용자 인터페이스 구축
                    </li>
                    <li className="flex items-center">
                      <span className="inline-block bg-blue-100 rounded-full w-2 h-2 mr-2"></span>
                      React Router - 페이지 라우팅
                    </li>
                    <li className="flex items-center">
                      <span className="inline-block bg-blue-100 rounded-full w-2 h-2 mr-2"></span>
                      Tailwind CSS - 스타일링
                    </li>
                    <li className="flex items-center">
                      <span className="inline-block bg-blue-100 rounded-full w-2 h-2 mr-2"></span>
                      Lucide React - 아이콘
                    </li>
                  </ul>
                </div>
                <div>
                  <h4 className="text-lg font-medium text-gray-900 mb-4">백엔드 (계획)</h4>
                  <ul className="space-y-2 text-gray-600">
                    <li className="flex items-center">
                      <span className="inline-block bg-blue-100 rounded-full w-2 h-2 mr-2"></span>
                      Node.js - 서버 환경
                    </li>
                    <li className="flex items-center">
                      <span className="inline-block bg-blue-100 rounded-full w-2 h-2 mr-2"></span>
                      Express - REST API
                    </li>
                    <li className="flex items-center">
                      <span className="inline-block bg-blue-100 rounded-full w-2 h-2 mr-2"></span>
                      OpenAI API - AI 기반 여행 계획 생성
                    </li>
                    <li className="flex items-center">
                      <span className="inline-block bg-blue-100 rounded-full w-2 h-2 mr-2"></span>
                      Azure - 클라우드 호스팅
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* 현재 상태 및 계획 */}
          <div className="mt-16">
            <h3 className="text-2xl font-bold text-gray-900 text-center mb-8">
              현재 상태 및 향후 계획
            </h3>
            <div className="bg-blue-50 rounded-lg p-8 border border-blue-100">
              <p className="text-blue-800 mb-4">
                현재 한국 AI 여행 플래너는 프론트엔드 MVP(최소 실행 가능 제품) 버전으로, 
                로컬 JSON 데이터와, 미리 정의된 일정 데이터를 활용하고 있습니다.
              </p>
              <h4 className="text-lg font-medium text-gray-900 mb-2">향후 계획:</h4>
              <ul className="space-y-2 text-gray-600 list-disc pl-5">
                <li>OpenAI API 연동하여 실제 AI 기반 여행 일정 생성 구현</li>
                <li>사용자 계정 시스템 및 여행 계획 저장 기능 추가</li>
                <li>지도 API 연동하여 위치 기반 서비스 강화</li>
                <li>실시간 교통 및 숙소 정보 업데이트 시스템 구축</li>
                <li>모바일 앱 버전 개발</li>
              </ul>
            </div>
          </div>

          {/* CTA */}
          <div className="mt-16 text-center">
            <Link
              to="/"
              className="inline-flex items-center justify-center px-5 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
            >
              지금 여행 계획하기
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutPage;