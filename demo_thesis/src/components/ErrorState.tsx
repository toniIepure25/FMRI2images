export function ErrorState({ message = 'Failed to load data' }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center mb-4">
        <span className="text-2xl text-red-400">!</span>
      </div>
      <p className="text-sm text-gray-400 mb-2">{message}</p>
      <p className="text-xs text-gray-600">Check that demo data files exist in the data/ directory</p>
    </div>
  );
}
