export default {
  expo: {
    name: "Pickleball Club",
    slug: "pickleball-club",
    scheme: "pickleball",
    version: "1.0.0",
    orientation: "portrait",
    platforms: ["ios", "android"],
    ios: { supportsTablet: false, bundleIdentifier: "com.pickleball.club" },
    android: { package: "com.pickleball.club" },
    extra: {
      apiUrl: process.env.EXPO_PUBLIC_API_URL,
      googleClientId: process.env.EXPO_PUBLIC_GOOGLE_CLIENT_ID,
    },
    plugins: ["expo-router"],
  },
};
