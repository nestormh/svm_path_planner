/**
 * 
 */
package sibtra.util;

/**
 * Thread para la ejecución cíclica. Al construirse arranca suspendido. 
 * Se puede {@link #activar()} o {@link #suspender()}.
 * En cada iteración se ejecuta método {@link #accion()}.
 * Evita el uso de {@link #suspend()} y {@link #resume()} que estan deprecados.
 * 
 * @author alberto
 *
 */
public abstract class ThreadSupendible extends Thread {

	private boolean suspendido=true;

	/** Arranca el thread suspendido */
	public ThreadSupendible() {
		start();
	}
	
	/** activa la ejecución cíclica */
	public final synchronized void activar() {
		if(suspendido) {
			suspendido=false;
			notify();
		}
	}

	/** Detine la ejecucion cíclica */
	public final synchronized void suspender() {
		if(!suspendido) {
			suspendido=true;
		}
	}

	/** @return si está suspendida la ejecución cíclica */
	public final synchronized  boolean isSuspendido() {
		return suspendido;
	}


	public final void run() {
		while(true) {
			accion();
			try {
				synchronized (this) {
					while (suspendido) wait(); 
				}
			} catch (InterruptedException e) {	}
		}
	}

	/** Método que se invoca en cada ciclo */
	protected abstract void accion();

}
