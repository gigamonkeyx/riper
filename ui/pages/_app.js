import '../styles/globals.css';
import { Special_Elite, Inter } from 'next/font/google';

const headline = Special_Elite({ subsets: ['latin'], weight: '400', variable: '--font-headline' });
const body = Inter({ subsets: ['latin'], weight: ['400','500','600','700'], variable: '--font-body' });

export default function App({ Component, pageProps }) {
  return (
    <div className={`${headline.variable} ${body.variable} font-body`}>
      <Component {...pageProps} />
    </div>
  );
} 