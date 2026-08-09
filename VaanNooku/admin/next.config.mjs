/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      { source: "/models", destination: "/" },
      { source: "/datasets", destination: "/" },
      { source: "/retraining", destination: "/" },
      { source: "/complaints", destination: "/" },
      { source: "/settings", destination: "/" },
    ];
  },
};

export default nextConfig;
