"use client";
import {
  Alert,
  AlertColor,
  Snackbar,
  SnackbarCloseReason,
} from "@mui/material";
import {
  createContext,
  PropsWithChildren,
  SyntheticEvent,
  useContext,
  useState,
} from "react";
type toastFunc = (msg: string, duration: number) => void;
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
  //   setMessage: (value: string) => void;
  //   setDuration: (value: number) => void;
  //   setOpen: (value: boolean) => void;
  //   setOnCloseHandler: (value: OnCloseHandler) => void;
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
type OnCloseHandler = (
  event: SyntheticEvent<any> | Event,
  reason: SnackbarCloseReason
) => void;
export function SnackBarProvider({ children }: PropsWithChildren) {
  const [open, setOpen] = useState(false);

  const [duration, setDuration] = useState(1000);
  const [message, setMessage] = useState("A message");
  const [msgType, setMsgType] = useState<AlertColor>("info");
  const handleClose: OnCloseHandler = (_, __) => {
    setOpen(false);
  };
  //   const [handleClose, setOnCloseHandler] = useState<OnCloseHandler>((_, __) => {
  //   });
  const toast = (type: AlertColor, m: string, d: number) => {
    setDuration(d);
    setMsgType(type);
    setMessage(m);
    setOpen(true);
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
        // setMessage,
        // setOpen,
        // setOnCloseHandler,
      }}
    >
      {children}
      <Snackbar
        open={open}
        autoHideDuration={duration}
        onClose={handleClose}
        message={message}
        // action={action}
      >
        <Alert severity={msgType}>{message}</Alert>
      </Snackbar>
    </snackBarContext.Provider>
  );
}
