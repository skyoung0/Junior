import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { MapPin, User, Menu, X, Search } from 'lucide-react';

const Header = ({ isLoggedIn, user, onLogout }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const navigate = useNavigate();

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/search?q=${encodeURIComponent(searchQuery)}`);
      setSearchQuery('');
      setMobileMenuOpen(false);
    }
  };

  return (
    <header className="bg-white shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* 로고 */}
          <div className="flex items-center">
            <Link to="/" className="flex-shrink-0 flex items-center">
              <MapPin className="h-8 w-8 text-blue-500" />
              <span className="ml-2 text-xl font-bold text-gray-900">한국 AI 여행 플래너</span>
            </Link>
          </div>

          {/* 검색창 (데스크톱) */}
          <div className="hidden md:flex items-center flex-1 px-8">
            <form onSubmit={handleSearch} className="w-full max-w-lg">
              <div className="relative">
                <input
                  type="text"
                  placeholder="여행지 검색..."
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-gray-400" />
                </div>
                <button
                  type="submit"
                  className="absolute inset-y-0 right-0 pr-3 flex items-center text-blue-500 hover:text-blue-600"
                >
                  검색
                </button>
              </div>
            </form>
          </div>
          
          {/* 데스크톱 메뉴 */}
          <div className="hidden md:flex items-center space-x-4">
            <Link to="/" className="text-gray-700 hover:text-blue-500 px-3 py-2">홈</Link>
            <Link to="/about" className="text-gray-700 hover:text-blue-500 px-3 py-2">소개</Link>
            
            {isLoggedIn ? (
              <div className="flex items-center space-x-4">
                <span className="text-gray-700">
                  <User className="inline h-4 w-4 mr-1" />
                  {user?.username || '사용자'}님
                </span>
                <button 
                  onClick={onLogout} 
                  className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded"
                >
                  로그아웃
                </button>
              </div>
            ) : (
              <Link to="/login" className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded">
                로그인
              </Link>
            )}
          </div>
          
          {/* 모바일 메뉴 버튼 */}
          <div className="flex md:hidden items-center">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="text-gray-700 hover:text-blue-500"
            >
              {mobileMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </div>
      
      {/* 모바일 메뉴 */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-white border-t">
          {/* 모바일 검색 */}
          <div className="px-4 py-3">
            <form onSubmit={handleSearch}>
              <div className="relative">
                <input
                  type="text"
                  placeholder="여행지 검색..."
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-gray-400" />
                </div>
              </div>
            </form>
          </div>
          
          <div className="px-2 pt-2 pb-3 space-y-1">
            <Link 
              to="/" 
              className="block px-3 py-2 text-gray-700 hover:bg-gray-100 hover:text-blue-500"
              onClick={() => setMobileMenuOpen(false)}
            >
              홈
            </Link>
            <Link 
              to="/about" 
              className="block px-3 py-2 text-gray-700 hover:bg-gray-100 hover:text-blue-500"
              onClick={() => setMobileMenuOpen(false)}
            >
              소개
            </Link>
            
            {isLoggedIn ? (
              <>
                <div className="px-3 py-2 text-gray-700">
                  <User className="inline h-4 w-4 mr-1" />
                  {user?.username || '사용자'}님
                </div>
                <button 
                  onClick={() => {
                    onLogout();
                    setMobileMenuOpen(false);
                  }} 
                  className="w-full text-left px-3 py-2 text-red-500 hover:bg-gray-100"
                >
                  로그아웃
                </button>
              </>
            ) : (
              <Link 
                to="/login" 
                className="block px-3 py-2 text-gray-700 hover:bg-gray-100 hover:text-blue-500"
                onClick={() => setMobileMenuOpen(false)}
              >
                로그인
              </Link>
            )}
          </div>
        </div>
      )}
    </header>
  );
};

export default Header;