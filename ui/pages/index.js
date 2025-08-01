import Head from 'next/head';
import TonasketSimUI from '../tonasket-sim-ui';

export default function Home() {
  return (
    <>
      <Head>
        <title>Tonasket Bakery Simulation - RIPER-Ω</title>
        <meta name="description" content="Interactive simulation for Tonasket Bakery with real-time parameter adjustment" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="true" />
        <link 
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" 
          rel="stylesheet" 
        />
      </Head>
      <TonasketSimUI />
    </>
  );
}
