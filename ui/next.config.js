/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  experimental: {
    appDir: false,
  },
  env: {
    SIMULATION_API_URL: process.env.SIMULATION_API_URL || 'http://localhost:8000',
    OLLAMA_API_URL: process.env.OLLAMA_API_URL || 'http://localhost:11434',
  },
  async rewrites() {
    return [
      {
        source: '/api/simulation/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ];
  },
  webpack: (config, { isServer }) => {
    // Fixes npm packages that depend on `fs` module
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
      };
    }
    return config;
  },
  images: {
    domains: ['localhost'],
  },
  poweredByHeader: false,
  compress: true,
  generateEtags: false,
  httpAgentOptions: {
    keepAlive: true,
  },
};

module.exports = nextConfig;
