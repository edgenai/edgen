'use client'

import Link from 'next/link'
import {
  type MotionValue,
  motion,
  useMotionTemplate,
  useMotionValue,
} from 'framer-motion'

import { GridPattern } from '@/components/GridPattern'
import { Heading } from '@/components/Heading'
import { ChatBubbleIcon } from '@/components/icons/ChatBubbleIcon'
import { MicrophoneIcon } from '@/components/icons/MicrophoneIcon'
import { CubeIcon } from '@/components/icons/CubeIcon'
import { EnvelopeIcon } from '@/components/icons/EnvelopeIcon'
import { UserIcon } from '@/components/icons/UserIcon'
import { UsersIcon } from '@/components/icons/UsersIcon'

interface Endpoint {
  href: string
  name: string
  description: string
  icon: React.ComponentType<{ className?: string }>
  pattern: Omit<
    React.ComponentPropsWithoutRef<typeof GridPattern>,
    'width' | 'height' | 'x'
  >
}

const endpoints: Array<Endpoint> = [
  {
    href: '/api-reference/audio',
    name: 'Audio',
    description: 'Convert audio to text or text to audio.',
    icon: MicrophoneIcon,
    pattern: {
      y: 16,
      squares: [
        [0, 1],
        [1, 3],
      ],
    },
  },
  {
    href: '/api-reference/chat',
    name: 'Chat',
    description:
      'Create a chat response, given a list of input chat messages.',
    icon: ChatBubbleIcon,
    pattern: {
      y: 16,
      squares: [
        [0, 1],
        [1, 3],
      ],
    },
  },
  {
    href: '/api-reference/models',
    name: 'Models',
    description:
      'Manage models: list local models, retrieve model information and remove models.',
    icon: CubeIcon,
    pattern: {
      y: 16,
      squares: [
        [0, 1],
        [1, 3],
      ],
    },
  },
]

function EndpointIcon({ icon: Icon }: { icon: Endpoint['icon'] }) {
  return (
    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-zinc-900/5 ring-1 ring-zinc-900/25 backdrop-blur-[2px] transition duration-300 group-hover:bg-white/50 group-hover:ring-zinc-900/25 dark:bg-white/7.5 dark:ring-white/15 dark:group-hover:bg-yellow-300/10 dark:group-hover:ring-yellow-400">
      <Icon className="h-8 w-8 fill-zinc-700/10 stroke-zinc-700 transition-colors duration-300 group-hover:stroke-zinc-900 dark:fill-white/10 dark:stroke-zinc-400 dark:group-hover:fill-yellow-300/10 dark:group-hover:stroke-yellow-400" />
    </div>
  )
}

function EndpointPattern({
  mouseX,
  mouseY,
  ...gridProps
}: Endpoint['pattern'] & {
  mouseX: MotionValue<number>
  mouseY: MotionValue<number>
}) {
  let maskImage = useMotionTemplate`radial-gradient(180px at ${mouseX}px ${mouseY}px, white, transparent)`
  let style = { maskImage, WebkitMaskImage: maskImage }

  return (
    <div className="pointer-events-none">
      <div className="absolute inset-0 rounded-2xl transition duration-300 [mask-image:linear-gradient(white,transparent)] group-hover:opacity-50">
        <GridPattern
          width={72}
          height={56}
          x="50%"
          className="absolute inset-x-0 inset-y-[-30%] h-[160%] w-full skew-y-[-18deg] fill-black/[0.02] stroke-black/5 dark:fill-white/1 dark:stroke-white/2.5"
          {...gridProps}
        />
      </div>
      <motion.div
        className="absolute inset-0 rounded-2xl bg-gradient-to-r from-[#F3D7AE] to-[#EDE7D7] opacity-0 transition duration-300 group-hover:opacity-100 dark:from-[#776942]"
        style={style}
      />
      <motion.div
        className="absolute inset-0 rounded-2xl opacity-0 mix-blend-overlay transition duration-300 group-hover:opacity-100"
        style={style}
      >
        <GridPattern
          width={72}
          height={56}
          x="50%"
          className="absolute inset-x-0 inset-y-[-30%] h-[160%] w-full skew-y-[-18deg] fill-black/50 stroke-black/70 dark:fill-white/2.5 dark:stroke-white/10"
          {...gridProps}
        />
      </motion.div>
    </div>
  )
}

function Endpoint({ endpoint }: { endpoint: Endpoint }) {
  let mouseX = useMotionValue(0)
  let mouseY = useMotionValue(0)

  function onMouseMove({
    currentTarget,
    clientX,
    clientY,
  }: React.MouseEvent<HTMLDivElement>) {
    let { left, top } = currentTarget.getBoundingClientRect()
    mouseX.set(clientX - left)
    mouseY.set(clientY - top)
  }

  return (
    <div
      key={endpoint.href}
      onMouseMove={onMouseMove}
      className="group relative flex rounded-2xl bg-zinc-50 transition-shadow hover:shadow-md hover:shadow-zinc-900/5 dark:bg-white/2.5 dark:hover:shadow-black/5"
    >
      <EndpointPattern {...endpoint.pattern} mouseX={mouseX} mouseY={mouseY} />
      <div className="absolute inset-0 rounded-2xl ring-1 ring-inset ring-zinc-900/7.5 group-hover:ring-zinc-900/10 dark:ring-white/10 dark:group-hover:ring-white/20" />
      <div className="relative rounded-2xl px-4 pb-4 pt-10">
        <EndpointIcon icon={endpoint.icon} />
        <h3 className="mt-4 text-lg font-semibold leading-7 text-zinc-900 dark:text-white">
          <Link href={endpoint.href}>
            <span className="absolute inset-0 rounded-2xl" />
            {endpoint.name}
          </Link>
        </h3>
        <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
          {endpoint.description}
        </p>
      </div>
    </div>
  )
}

export function Endpoints() {
  return (
    <div className="my-16 xl:max-w-none">
      <Heading level={2} id="endpoints">
        Endpoints
      </Heading>
      <div className="not-prose mt-4 grid grid-cols-1 gap-8 border-t border-zinc-900/5 pt-10 dark:border-white/5 sm:grid-cols-2 xl:grid-cols-4">
        {endpoints.map((endpoint) => (
          <Endpoint key={endpoint.href} endpoint={endpoint} />
        ))}
      </div>
    </div>
  )
}
