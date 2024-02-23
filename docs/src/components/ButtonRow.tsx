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
            <ul className="flex w-fit list-none p-0.5 mt-6 mb-0.5 h-10 rounded-lg bg-yellow-400">
                {types.map((t, i) => {
                return (
                    <li
                        key={i}
                        className={clsx(
                            `${types[currentType] == t ? 'bg-yellow-400' : 'bg-yellow-100'}`,
                            'flex h-full text-black justify-center items-center',
                            'first:rounded-bl-lg first:rounded-tl-lg last:rounded-tr-lg last:rounded-br-lg',
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
