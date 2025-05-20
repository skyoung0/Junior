import React from 'react';
import { ExternalLink } from 'lucide-react';
import { formatPrice } from '../utils/helpers';

const HotelCard = ({ hotel }) => {
  return (
    <div className="bg-white overflow-hidden shadow-lg rounded-lg flex flex-col h-full">
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
            <span className="text-gray-900 font-medium">1박 {formatPrice(hotel.pricePerNight)}</span>
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
  );
};

export default HotelCard;