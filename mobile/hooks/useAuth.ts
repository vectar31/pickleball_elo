import { useEffect } from 'react';
import * as SecureStore from 'expo-secure-store';
import { useAuthStore } from '@/stores/authStore';

const AUTH_TOKEN_KEY = 'auth_token';
const USER_ID_KEY = 'user_id';

export function useAuth() {
  const { token, userId, hydrated, setAuth, clearAuth, setHydrated } = useAuthStore();

  useEffect(() => {
    async function hydrate() {
      const [storedToken, storedUserId] = await Promise.all([
        SecureStore.getItemAsync(AUTH_TOKEN_KEY),
        SecureStore.getItemAsync(USER_ID_KEY),
      ]);
      if (storedToken && storedUserId) {
        setAuth(storedToken, storedUserId);
      }
      setHydrated();
    }
    if (!hydrated) hydrate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function login(token: string, userId: string) {
    await Promise.all([
      SecureStore.setItemAsync(AUTH_TOKEN_KEY, token),
      SecureStore.setItemAsync(USER_ID_KEY, userId),
    ]);
    setAuth(token, userId);
  }

  async function logout() {
    await Promise.all([
      SecureStore.deleteItemAsync(AUTH_TOKEN_KEY),
      SecureStore.deleteItemAsync(USER_ID_KEY),
    ]);
    clearAuth();
  }

  return { token, userId, hydrated, isAuthenticated: !!token, login, logout };
}
