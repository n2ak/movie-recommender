"use client";
import { ThemeOptions, PaletteMode } from "@mui/material/styles";
import { getDesignTokens } from "./themePrimitives";
// import {
//   inputsCustomizations,
//   dataDisplayCustomizations,
//   feedbackCustomizations,
//   navigationCustomizations,
//   surfacesCustomizations,
// } from "./customizations";

export default function getTheme(mode: PaletteMode): ThemeOptions {
  return {
    ...getDesignTokens(mode),
    // components: {
    //   ...inputsCustomizations,
    //   ...dataDisplayCustomizations ,
    //   ...feedbackCustomizations,
    //   ...navigationCustomizations,
    //   ...surfacesCustomizations,
    // },
  };
}
