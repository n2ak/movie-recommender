"use client";
import { Alert, AlertColor, Snackbar } from "@mui/material";
import { createContext, PropsWithChildren, useContext, useState } from "react";

type toastFunc = (msg: string, duration?: number) => void;

interface SnackBarContextInterface {
  success: toastFunc;
  info: toastFunc;
  warning: toastFunc;
  error: toastFunc;
  handlePromise: <T>(
    p: Promise<T>,
    onSucc: string | undefined,
    onErr?: string | undefined
  ) => Promise<T | null>;
}
const defaultDuration = 2000;
const snackBarContext = createContext<SnackBarContextInterface | undefined>(
  undefined
);

export function useSnackBar() {
  const context = useContext(snackBarContext);
  if (context === undefined) {
    throw new Error("useCookieContext must be used within a CookieProvider");
  }
  return context;
}

export function SnackBarProvider({ children }: PropsWithChildren) {
  const [state, setState] = useState<{
    open: boolean;
    duration: number;
    message: string;
    msgType: AlertColor;
  }>({
    open: false,
    duration: 1000,
    message: "",
    msgType: "info",
  });
  const toast = (type: AlertColor, m: string, d?: number) => {
    setState({
      duration: d || defaultDuration,
      msgType: type,
      message: m,
      open: true,
    });
  };
  return (
    <snackBarContext.Provider
      value={{
        success: (m, d) => toast("success", m, d),
        info: (m, d) => toast("info", m, d),
        warning: (m, d) => toast("warning", m, d),
        error: (m, d) => toast("error", m, d),
        handlePromise: async function handleRequestToast<T>(
          promise: Promise<T>,
          onSucc: string | undefined,
          onErr: string | undefined = undefined
        ) {
          try {
            const t = await promise;
            if (onSucc !== undefined) this.success(onSucc, defaultDuration);
            return t;
          } catch (e) {
            if (onErr !== undefined) this.error(onErr, defaultDuration);
            return null;
          }
        },
        // setDuration,
      }}
    >
      {children}
      <Snackbar
        open={state.open}
        autoHideDuration={state.duration}
        onClose={() => {
          setState({
            ...state,
            open: false,
          });
        }}
        message={state.message}
        className="z-20"
      >
        <Alert severity={state.msgType} className="text-sm font-medium">
          {state.message}
        </Alert>
      </Snackbar>
    </snackBarContext.Provider>
  );
}
