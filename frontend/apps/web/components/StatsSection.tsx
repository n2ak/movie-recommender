import { User } from "next-auth";

export default function StatsSection({ user }: { user: User }) {
  return (
    <>
      <h1 className="text-2xl font-semibold mb-4">Settings</h1>
      <div className="bg-white p-4 rounded-lg shadow-sm">Stats</div>
    </>
  );
}
