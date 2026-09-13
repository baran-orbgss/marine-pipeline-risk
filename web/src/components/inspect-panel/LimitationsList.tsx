import styles from './LimitationsList.module.css'

export function LimitationsList({ limitations }: { limitations: string[] }) {
  if (limitations.length === 0) return null
  return (
    <div className={styles.wrapper}>
      <div className={styles.title}>Limitations</div>
      <ul className={styles.list}>
        {limitations.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
    </div>
  )
}
