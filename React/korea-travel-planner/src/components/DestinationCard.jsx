import React from 'react';
import { useNavigate } from 'react-router-dom';
import { MapPin } from 'lucide-react';

const DestinationCard = ({ destination }) => {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate(`/results/${destination.name}`);
  };

  return (
    <div className="bg-white overflow-hidden shadow-lg rounded-lg">
      <div className="relative">
        <img 
          src={destination.image} 
          alt={destination.name} 
          className="w-full h-48 object-cover"
        />
        <div className="absolute top-0 left-0 bg-blue-500 text-white px-3 py-1 m-3 rounded-full flex items-center">
          <MapPin className="h-4 w-4 mr-1" />
          <span className="text-sm font-medium">{destination.name}</span>
        </div>
      </div>
      <div className="p-6">
        <h3 className="text-xl font-semibold text-gray-900">{destination.name}</h3>
        <p className="mt-2 text-gray-600">{destination.shortDescription || destination.description}</p>
        
        <div className="mt-4 flex flex-wrap gap-2">
          {destination.tags && destination.tags.map((tag, index) => (
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
            onClick={handleClick}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            여행 계획 만들기
          </button>
        </div>
      </div>
    </div>
  );
};

export default DestinationCard;