import React, { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import { MapPin, User, Menu, X } from 'lucide-react';

// 페이지 컴포넌트 임포트
import HomePage from './pages/HomePage';
import ResultPage from './pages/ResultPage';
import AboutPage from './pages/AboutPage';
import LoginPage from './pages/LoginPage';

// 컴포넌트 임포트
import Header from './components/Header';
import Footer from './components/Footer';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState(null);

  // 로컬 스토리지에서 로그인 정보 확인
  React.useEffect(() => {
    const storedLoggedIn = localStorage.getItem('isLoggedIn') === 'true';
    const storedUser = JSON.parse(localStorage.getItem('user'));
    
    if (storedLoggedIn && storedUser) {
      setIsLoggedIn(true);
      setUser(storedUser);
    }
  }, []);

  // 간단한 로그인 처리 (실제 인증은 구현하지 않음)
  const handleLogin = (credentials) => {
    if (credentials.username === 'iksang' && credentials.password === '1234') {
      const userData = { username: credentials.username };
      setIsLoggedIn(true);
      setUser(userData);
      
      // 로컬 스토리지에 로그인 정보 저장
      localStorage.setItem('isLoggedIn', 'true');
      localStorage.setItem('user', JSON.stringify(userData));
      
      return true;
    }
    return false;
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUser(null);
    
    // 로컬 스토리지에서 로그인 정보 삭제
    localStorage.removeItem('isLoggedIn');
    localStorage.removeItem('user');
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* 헤더 */}
      <Header isLoggedIn={isLoggedIn} user={user} onLogout={handleLogout} />

      {/* 메인 콘텐츠 */}
      <main className="flex-grow">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/results/:destination" element={<ResultPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/login" element={<LoginPage onLogin={handleLogin} />} />
        </Routes>
      </main>

      {/* 푸터 */}
      <Footer />
    </div>
  );
}

export default App;