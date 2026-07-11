/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      { source: "/login", destination: "/" },
      { source: "/register", destination: "/" },
      { source: "/dashboard", destination: "/" },
      { source: "/analytics", destination: "/" },
      { source: "/inventory", destination: "/" },
      { source: "/products", destination: "/" },
      { source: "/suppliers", destination: "/" },
      { source: "/predictions", destination: "/" },
      { source: "/history", destination: "/" },
      { source: "/settings", destination: "/" },
    ];
  },
};

export default nextConfig;
