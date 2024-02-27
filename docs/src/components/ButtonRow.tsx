'use client'

import clsx from 'clsx'
import React, { useState } from 'react'
import { Col } from './mdx'

export function ButtonRow({
    types,
    children 
}: {
    types: string[]
    children: React.ReactNode 
}) {
  const [currentType, setCurrentType] = useState<number>(0)

  const onlyCurrent = (children: React.ReactNode ) => (
    <div>
      {React.Children.toArray(children).map((child, i) => i == currentType ? <div key={i}>{child}</div> : <div key={i}></div>)}
    </div>
  );

  return (
    <>
        <div className='flex h-fit flex-col items-end'>
            <ul className="flex w-fit list-none p-1.5 mt-6 mb-0.5 h-11 rounded-full bg-zinc-900 dark:bg-yellow-400/10 dark:bg-yellow-400/10 dark:ring-1 dark:ring-inset dark:ring-yellow-400/20 dark:hover:bg-yellow-400/10">
                {types.map((t, i) => {
                return (
                    <li
                        key={i}
                        className={clsx(
                            `${types[currentType] == t ? 'bg-yellow-300 dark:bg-yellow-100 text-black hover:bg-yellow-300' : 'bg-none text-white dark:text-yellow-400 dark:hover:text-yellow-300'}`,
                            'flex h-full justify-center items-center',
                            'first:rounded-full last:rounded-full',
                            'm-0 px-4',
                            'cursor-pointer'
                        )}
                        onClick={() => setCurrentType(i)}
                        >
                        <div className='w-fit h-fit'>
                            {t}
                        </div>
                    </li>
                )
                })}
            </ul>
        </div>
        <Col>
            {onlyCurrent(children)}
        </Col>
    </>
  )
}
