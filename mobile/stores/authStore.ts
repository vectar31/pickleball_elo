import { create } from 'zustand';

interface AuthState {
  token: string | null;
  userId: string | null;
  hydrated: boolean;
  setAuth: (token: string, userId: string) => void;
  clearAuth: () => void;
  setHydrated: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  token: null,
  userId: null,
  hydrated: false,
  setAuth: (token, userId) => set({ token, userId }),
  clearAuth: () => set({ token: null, userId: null }),
  setHydrated: () => set({ hydrated: true }),
}));
