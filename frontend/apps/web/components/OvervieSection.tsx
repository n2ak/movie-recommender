import { UserInfo } from "@/hooks/useAuthStore";
import { formatDate } from "@/lib/utils";

export default function OverviewSection({ user }: { user: UserInfo }) {
  return (
    <>
      <h1 className="text-2xl font-semibold mb-4">Profile Overview</h1>
      <div className="bg-white p-4 rounded-lg shadow-sm">
        {user.username && (
          <div className="text-lg font-bold mb-2">{user.username}</div>
        )}

        <div className="mt-4 space-y-2 text-sm text-gray-600">
          <div>
            <span className="font-medium">Joined:</span>{" "}
            {formatDate(user.createdAt)}
          </div>
          {/* Email */}
          {user.email && (
            <div>
              <span className="font-medium">Email:</span> {user.email}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
