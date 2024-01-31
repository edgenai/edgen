import Image from 'next/image'

import { Button } from '@/components/Button'
import { Heading } from '@/components/Heading'
import logoGo from '@/images/logos/go.svg'
import logoNode from '@/images/logos/node.svg'
import logoPython from '@/images/logos/python.svg'
import logoRust from '@/images/logos/rust.svg'

const apiclients = [
  {
    href: 'https://github.com/edgenai/edgen-client-python',
    name: 'Python',
    description:
      'Python is a programming language that lets you work quickly and integrate systems more effectively.',
    logo: logoPython,
  },
  {
    href: 'https://github.com/edgenai/edgen-client-node',
    name: 'Node.js',
    description:
      'Node.js® is an open-source, cross-platform JavaScript runtime environment.',
    logo: logoNode,
  },
  // {
  //   href: '#',
  //   name: 'Rust',
  //   description:
  //     'An open-source programming language that prioritizes memory safety and high performance.',
  //   logo: logoRust,
  // },
  // {
  //   href: '#',
  //   name: 'Go',
  //   description:
  //     'An open-source programming language supported by Google with built-in concurrency.',
  //   logo: logoGo,
  // },
]

export function APIClients() {
  return (
    <div className="my-16 xl:max-w-none">
      <Heading level={2} id="official-apiclients">
        Official API Clients
      </Heading>
      <div className="not-prose mt-4 grid grid-cols-1 gap-x-6 gap-y-10 border-t border-zinc-900/5 pt-10 dark:border-white/5 sm:grid-cols-2 xl:max-w-none xl:grid-cols-3">
        {apiclients.map((library) => (
          <div key={library.name} className="flex flex-row-reverse gap-6">
            <div className="flex-auto">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
                {library.name}
              </h3>
              <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
                {library.description}
              </p>
              <p className="mt-4">
                <Button href={library.href} variant="text" arrow="right">
                  Read more
                </Button>
              </p>
            </div>
            <Image
              src={library.logo}
              alt=""
              className="h-12 w-12"
              unoptimized
            />
          </div>
        ))}
      </div>
    </div>
  )
}
