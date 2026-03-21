import { create } from 'zustand';

interface CommunityState {
  activeCommunityId: string | null;
  setActiveCommunity: (id: string) => void;
}

export const useCommunityStore = create<CommunityState>((set) => ({
  activeCommunityId: null,
  setActiveCommunity: (id) => set({ activeCommunityId: id }),
}));
