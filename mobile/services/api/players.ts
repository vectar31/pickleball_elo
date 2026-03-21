import { apiClient } from '@/lib/apiClient';

export async function getPlayers(): Promise<string[]> {
  const res = await apiClient.get('/players');
  return res.data;
}
