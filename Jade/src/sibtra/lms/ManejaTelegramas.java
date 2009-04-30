/**
 * 
 */
package sibtra.lms;

/**
 * @author alberto
 *
 */
public interface ManejaTelegramas {

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
	 * @return mensaje contenido en telegrama
	 */
	public abstract byte[] LeeMensaje();

	/**
	 * Construye a telegarama a partir del mensaje pasado y lo envía.
	 * Espera la confirmación de que ha sido bien recibido.
	 * 
	 * @return true si se recibe confirmación
	 */
	public abstract boolean EnviaMensaje(byte[] mensaje);
	
	/**
	 * Cierra el puerto
	 */
	public boolean cierraPuerto();


}