/**
 * 
 */
package sibtra.lms;

/**
 * @author alberto
 *
 */
public abstract class ManejaTelegramas {
	
	protected int milisTOutDefecto = 200;

	/** Fija el time out por defecto (en milisegundos) */
	public void setDefaultTimeOut(int milisg) {
		if(milisg<0)
			throw new IllegalArgumentException("El time out debe ser número >=0");
		milisTOutDefecto=milisg;
	}
	
	/** @return el time out por defecto (en milisegundos) */
	public int getDefaultTimeOut() {
		return milisTOutDefecto;
	}

	/**
	 * Para abrir y configurar puerto serie
	 * 
	 * @param NombrePuerto nombre del puerto serie a utilizar
	 */
	public abstract boolean ConectaPuerto(String NombrePuerto);

	/** @return si el pueto ha sido correctamente inicializado por {@link #ConectaPuerto(String)} */
	public abstract boolean isInicializado();


	/** 
	 * Trata de fijar la velocidad de transmisisón del puerto al indicado
	 * @param baudrate velocidad desead
	 * @return ture si la velocidad es valida y se consiguió el cambio.
	 */
	public abstract boolean setBaudrate(int baudrate);

	/**
	 * Espera la llegada de un telegrama por la serial y extrae el mensaje
	 * @param milisTOut especifica milisegundos para dar <i>time out</i>. 
	 *   0 significa no time out.
	 * @return mensaje contenido en telegrama
	 */
	public abstract byte[] LeeMensaje(int milisTOut);

	/**
	 * Ídem que {@link #LeeMensaje(int)} con TimeOut por defecto 
	 */
	public byte[] LeeMensaje() {
		return LeeMensaje(milisTOutDefecto);
	}

	/**
	 * Construye a telegarama a partir del mensaje pasado y lo envía.
	 * Espera la confirmación de que ha sido bien recibido.
	 * @param mensaje a enviar
	 * @param milisTOut especifica milisegundos para dar <i>time out</i>. 
	 *   en la espera de confirmación. 0 significa no time out.
	 * @return true si se recibe confirmación
	 */
	public boolean EnviaMensaje(byte[] mensaje, int milisTOut) {
		return EnviaMensajeSinConfirmacion(mensaje) && esperaConfirmacion(milisTOut);
	}

	/**
	 * Ídem {@link #EnviaMensaje(byte[], int)} con TimeOut por defecto
	 */
	public boolean EnviaMensaje(byte[] mensaje) {
		return EnviaMensaje(mensaje, milisTOutDefecto);
	}
	
	/**
	 * Construye a telegarama a partir del mensaje pasado y lo envía.
	 * NO espera la confirmación de que ha sido bien recibido.
 	 * @param mensaje a enviar
 	 * @return true si se evió correctamente 
	 */
	public abstract boolean EnviaMensajeSinConfirmacion(byte[] mensaje);

	/** Espera la confirmación
	 * @param milisTOut especifica milisegundos para dar <i>time out</i>.
	 * @return true si se recibe confirmación
	 */
	public abstract boolean esperaConfirmacion(int milisTOut);

	/** Vacia todo el buffer de entrada de la serial */
	public abstract void purgaBufferEntrada();

	
	/**
	 * Cierra el puerto
	 */
	public abstract boolean cierraPuerto();


}