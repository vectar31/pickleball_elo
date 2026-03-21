import axios from 'axios';

const BASE_URL = process.env.EXPO_PUBLIC_API_URL;

export async function loginWithPassword(username: string, password: string) {
  const params = new URLSearchParams();
  params.append('username', username);
  params.append('password', password);
  params.append('grant_type', 'password');

  const res = await axios.post(`${BASE_URL}/auth/login`, params.toString(), {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  });
  return res.data as { access_token: string; token_type: string };
}
