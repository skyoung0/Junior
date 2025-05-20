import React from 'react';
import { Clock } from 'lucide-react';

const ItineraryDay = ({ day }) => {
  return (
    <div className="mb-8 border rounded-lg overflow-hidden shadow-sm">
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
  );
};

export default ItineraryDay;