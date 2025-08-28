module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      // Fix: Use react-native-worklets instead of react-native-reanimated
      'react-native-worklets/plugin',
      'react-native-reanimated/plugin', // This should be last
    ],
  };
};