import pino from "pino";
import pretty from "pino-pretty";

const stream = pretty({
  levelFirst: true,
  colorize: true,
  ignore: "time,hostname,pid",
  timestampKey: "timestamp",
});

const logger = pino(
  {
    level: "debug",
    formatters: {
      level: (level) => ({ level }),
    },
    timestamp: true,
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
