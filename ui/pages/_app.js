import '../styles/globals.css';
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'], weight: ['400','500','600','700'] });

export default function App({ Component, pageProps }) {
  return (
    <div className={`${inter.className} font-sans`}>
      <Component {...pageProps} />
    </div>
  );
}