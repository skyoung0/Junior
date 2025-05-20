import React from 'react';
import { Link } from 'react-router-dom';
import { MapPin, ExternalLink } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-gray-800 text-white py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="md:flex md:justify-between">
          <div className="mb-8 md:mb-0">
            <Link to="/" className="flex items-center">
              <MapPin className="h-8 w-8 text-blue-400" />
              <span className="ml-2 text-xl font-bold">한국 AI 여행 플래너</span>
            </Link>
            <p className="mt-2 text-gray-400">한국 여행의 모든 것을 AI가 도와드립니다</p>
          </div>
          <div className="grid grid-cols-2 gap-8 md:gap-16">
            <div>
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">사이트 링크</h3>
              <ul className="mt-4 space-y-2">
                <li><Link to="/" className="text-gray-300 hover:text-white">홈</Link></li>
                <li><Link to="/about" className="text-gray-300 hover:text-white">소개</Link></li>
                <li><Link to="/login" className="text-gray-300 hover:text-white">로그인</Link></li>
              </ul>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">외부 링크</h3>
              <ul className="mt-4 space-y-2">
                <li>
                  <a 
                    href="https://www.letskorail.com/" 
                    target="_blank" 
                    rel="noopener noreferrer" 
                    className="text-gray-300 hover:text-white flex items-center"
                  >
                    기차 예약 <ExternalLink className="ml-1 h-3 w-3" />
                  </a>
                </li>
                <li>
                  <a 
                    href="https://www.goodchoice.kr/" 
                    target="_blank" 
                    rel="noopener noreferrer" 
                    className="text-gray-300 hover:text-white flex items-center"
                  >
                    숙소 예약 <ExternalLink className="ml-1 h-3 w-3" />
                  </a>
                </li>
                <li>
                  <a 
                    href="https://korean.visitkorea.or.kr/" 
                    target="_blank" 
                    rel="noopener noreferrer" 
                    className="text-gray-300 hover:text-white flex items-center"
                  >
                    대한민국 관광공사 <ExternalLink className="ml-1 h-3 w-3" />
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </div>
        <div className="mt-8 border-t border-gray-700 pt-8 md:flex md:items-center md:justify-between">
          <p className="text-base text-gray-400">&copy; 2025 한국 AI 여행 플래너. 모든 권리 보유.</p>
          <div className="mt-4 md:mt-0">
            <p className="text-gray-400">이 사이트는 데모용입니다.</p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;