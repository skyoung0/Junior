import React, { useState } from 'react';
import { Train, ExternalLink } from 'lucide-react';
import { formatPrice } from '../utils/helpers';

const TrainSchedule = ({ trainData, destination }) => {
  const [departureCity, setDepartureCity] = useState(
    trainData && trainData.length > 0 ? trainData[0].from : '서울'
  );

  // 제주도는 기차가 없으므로 특별 처리
  if (destination === '제주도') {
    return (
      <div className="bg-yellow-50 p-6 rounded-lg">
        <h3 className="flex items-center text-lg font-medium text-yellow-800 mb-4">
          <Train className="h-5 w-5 mr-2" /> 
          항공편 안내
        </h3>
        <p className="text-yellow-700 mb-4">제주도는 기차로 갈 수 없는 섬입니다. 항공편을 이용해주세요.</p>
        <a 
          href="https://www.koreanair.com" 
          target="_blank" 
          rel="noopener noreferrer"
          className="inline-flex items-center text-blue-600 hover:text-blue-800"
        >
          항공권 예약하기 <ExternalLink className="ml-1 h-4 w-4" />
        </a>
      </div>
    );
  }

  // 데이터가 없는 경우
  if (!trainData || trainData.length === 0) {
    return (
      <div className="bg-gray-50 p-6 rounded-lg">
        <p className="text-gray-600">현재 {destination}으로 가는 기차 정보가 없습니다.</p>
      </div>
    );
  }

  const filteredRoute = trainData.find(route => route.from === departureCity);

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="px-6 py-4 bg-blue-50">
        <h3 className="flex items-center text-lg font-medium text-gray-900">
          <Train className="h-5 w-5 mr-2 text-blue-500" /> 
          {destination}행 기차 시간표
        </h3>
      </div>
      
      <div className="p-6">
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

        {filteredRoute && filteredRoute.departures && filteredRoute.departures.length > 0 ? (
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
                {filteredRoute.departures.map((train, idx) => (
                  <tr key={idx}>
                    <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6">
                      {filteredRoute.trainType}
                    </td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{train.time}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{train.duration}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{formatPrice(train.price)}</td>
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
        
        <div className="mt-6 text-center">
          <a 
            href="https://www.letskorail.com/" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            코레일 홈페이지 방문하기 <ExternalLink className="ml-1 h-4 w-4" />
          </a>
        </div>
      </div>
    </div>
  );
};

export default TrainSchedule;