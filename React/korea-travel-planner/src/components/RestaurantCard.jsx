import React from 'react';
import { Map } from 'lucide-react';

const RestaurantCard = ({ restaurant }) => {
  return (
    <div className="bg-white overflow-hidden shadow-lg rounded-lg flex flex-col h-full">
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
          {restaurant.mustTry && restaurant.mustTry.length > 0 && (
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
  );
};

export default RestaurantCard;