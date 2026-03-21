import { apiClient } from '@/lib/apiClient';

export interface SinglesMatch {
  date: string;       // "YYYY-MM-DD"
  player1: string;
  player2: string;
  score1: number;
  score2: number;
}

export async function getSinglesMatches(): Promise<SinglesMatch[]> {
  const res = await apiClient.get('/matches/singles');
  return res.data;
}

export async function postSinglesMatch(match: SinglesMatch): Promise<void> {
  await apiClient.post('/matches/singles', match);
}
