export function ErrorState({ message = 'Failed to load data' }: { message?: string }) {
  return (
    <div className="premium-page-bg flex min-h-[60vh] flex-col items-center justify-center px-6 py-20 text-center">
      <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-status-error/20 bg-status-error/10">
        <span className="text-xl font-semibold text-status-error">!</span>
      </div>
      <p className="mb-2 text-sm font-semibold text-text-primary">{message}</p>
      <p className="max-w-md text-xs leading-relaxed text-text-muted">
        Check that demo data files exist in the data directory and that any live backend artifacts are available.
      </p>
    </div>
  );
}
