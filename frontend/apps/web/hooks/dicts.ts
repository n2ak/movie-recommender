export const englishDict = {
  searchAMovie: "Search for a movie...",
  signIn: "Sign in",
  home: "Home",
  settings: "Settings",
  overview: "Overview",
  save: "Save",
  language: "Language",
  moviesWeThingYouWouldLike: "Movies we thing you would like:",
  recommendedMovies: (v: string) => `Recommended '${v}' movies:`,
  darkMode: "Dark mode",
  signOut: "Sign out",
};
type Dict = typeof englishDict;

// export const frenchDict: {
//   [key in keyof Dict]: string;
// } = {
//   searchAMovie: "Rechercher un film...",
//   signIn: "Se connecter",
//   home: "Accueil",
//   settings: "Parametres",
//   overview: "Aperçu",
// darkMode: "Mode sombre",
// signOut:"Se déconnecter",
// save:"Enregistrer"
// };
