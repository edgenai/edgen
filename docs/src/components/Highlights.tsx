import { Button } from '@/components/Button'
import { Heading } from '@/components/Heading'

const highlights = [
  {
    href: '/guides/quickstart',
    name: 'Quickstart',
    description: 'Learn how to authenticate your API requests.',
  },
  {
    href: '/documentation/configuration',
    name: 'Configuration',
    description: 'Understand how to work with paginated responses.',
  },
  {
    href: '/webhooks',
    name: 'Webhooks',
    description:
      'Learn how to programmatically configure webhooks for your app.',
  },
]

export function Highlights() {
  return (
    <div className="my-16 xl:max-w-none">
      <Heading level={2} id="highlights">
        Highlights
      </Heading>
      <div className="not-prose mt-4 grid grid-cols-1 gap-8 border-t border-zinc-900/5 pt-10 dark:border-white/5 sm:grid-cols-2 xl:grid-cols-4">
        {highlights.map((guide) => (
          <div key={guide.href}>
            <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
              {guide.name}
            </h3>
            <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
              {guide.description}
            </p>
            <p className="mt-4">
              <Button href={guide.href} variant="text" arrow="right">
                Read more
              </Button>
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
