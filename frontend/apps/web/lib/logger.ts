import pino from "pino";
import pretty from "pino-pretty";

const stream = pretty({
  levelFirst: true,
  colorize: true,
  ignore: "time,hostname,pid",
});

const logger = pino(
  {
    level: "debug",
    messageKey: "dad",
    formatters: {
      level: (level) => ({ level }),
    },
  },
  process.env.NODE_ENV === "production" ? undefined : stream
);

export const startTimer = () => {
  const start = process.hrtime.bigint();
  return () => {
    const end = process.hrtime.bigint();
    const duration = Math.floor(Number(end - start) / 1_000_000);
    return duration;
  };
};

export default logger;
