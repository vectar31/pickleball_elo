import { defineConfig } from 'orval';

export default defineConfig({
  pickleballApi: {
    input: '../docs/api-contracts/openapi.json',
    output: {
      target: './services/api/index.ts',
      client: 'react-query',
      baseUrl: process.env.EXPO_PUBLIC_API_URL,
    },
  },
});
