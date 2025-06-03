import type { UserInfo } from "@/hooks/useAuthStore";
import { formatDate } from "@/lib/utils";

export default function OverviewSection({ user }: { user: UserInfo }) {
  return (
    <div className="">
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
  );
}
