/** @type {import('tailwindcss').Config} */
export const content = [
    "./src/**/*.{js,jsx,ts,tsx}",
];
export const theme = {
    extend: {
        colors: {
            // 커스텀 색상 - 한국 전통 색상 팔레트
            'traditional': {
                'blue': '#1A63A8', // 청자색
                'red': '#CB2F2A', // 홍색
                'yellow': '#F0CA00', // 황색
                'white': '#F8F8F8', // 백색
                'black': '#212121', // 흑색
            },
            // 그라데이션용 색상
            'gradient': {
                'start': '#3B82F6', // 시작 색상 (청색)
                'end': '#1E40AF', // 종료 색상 (짙은 청색)
            },
        },
        fontFamily: {
            'sans': ['Pretendard', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
        },
        backgroundImage: {
            'hero-pattern': "url('https://images.unsplash.com/photo-1538485399081-7a66562c35fc?q=80&w=1974&auto=format&fit=crop')",
        },
        animation: {
            'fade-in': 'fadeIn 0.3s ease-in-out',
            'slide-up': 'slideUp 0.4s ease-out',
        },
        keyframes: {
            fadeIn: {
                '0%': { opacity: '0' },
                '100%': { opacity: '1' },
            },
            slideUp: {
                '0%': { transform: 'translateY(20px)', opacity: '0' },
                '100%': { transform: 'translateY(0)', opacity: '1' },
            },
        },
        boxShadow: {
            'card': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            'card-hover': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        },
    },
};
export const plugins = [
    // eslint-disable-next-line no-undef
    require('@tailwindcss/forms'), // 폼 스타일링
];