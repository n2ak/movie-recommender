export default function OverviewSection({ user }: any) {
  return (
    <>
      <h1 className="text-2xl font-semibold mb-4">Profile Overview</h1>
      <div className="bg-white p-4 rounded-lg shadow-sm">
        <p className="text-gray-700">{user.bio}</p>

        <div className="mt-4 space-y-2 text-sm text-gray-600">
          <div>
            <span className="font-medium">Location:</span> {user.location}
          </div>
          <div>
            <span className="font-medium">Website:</span>{" "}
            <a
              href={user.website}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              {user.website}
            </a>
          </div>
          <div>
            <span className="font-medium">Joined:</span> {user.joined}
          </div>
        </div>
      </div>
    </>
  );
}
